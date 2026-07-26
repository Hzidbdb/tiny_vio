/*
 * @Author: garygoo
 * @Date: 2026-07-25
 * @Description: Levenberg-Marquardt 非线性优化求解器
 *   依赖: tiny_linalg (mat_cholesky, mat_cholesky_solve, vec_dot, vec_L2)
 */

#include "stdio.h"
#include "math.h"
#include "tiny_opt.h"
#include "../tiny_linalg/tiny_linalg.h"

/* ---------- 默认配置 ---------- */

static lm_config_t default_config(void)
{
    lm_config_t cfg;
    cfg.max_iter    = 100;
    cfg.tau         = 1e-3;
    cfg.eps_grad    = 1e-8;
    cfg.eps_step    = 1e-8;
    cfg.eps_cost    = 1e-8;
    cfg.lambda_init = -1.0;
    return cfg;
}

/* ---------- 数值 Jacobian (前向差分) ---------- */

static void numerical_jacobian(double *x, void *data,
                                lm_residual_t residual,
                                int m, int n,
                                double *J, const double *r_center)
{
    double r_pert[LM_MAX_RESIDUALS];

    for (int j = 0; j < n; j++) {
        /* h = sqrt(eps) * max(|x_j|, 1.0), eps ≈ 2.22e-16 */
        double h = 1.4901161193847656e-8 * fmax(fabs(x[j]), 1.0);
        double saved = x[j];
        x[j] = saved + h;
        residual(x, data, r_pert, m, n);
        x[j] = saved;

        double inv_h = 1.0 / h;
        for (int i = 0; i < m; i++)
            J[i * n + j] = (r_pert[i] - r_center[i]) * inv_h;
    }
}

/* ---------- 法方程构建: H = J^T J, g = J^T r ---------- */

static void build_normal_eqns(const double *J, const double *r,
                               int m, int n, double *H, double *g)
{
    /* H[i][j] = Σ_k J[k*n+i] * J[k*n+j] */
    /* g[i]   = Σ_k J[k*n+i] * r[k]        */
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++)
                s += J[k * n + i] * J[k * n + j];
            H[i * n + j] = s;
            H[j * n + i] = s;   /* 对称 */
        }
        double sg = 0.0;
        for (int k = 0; k < m; k++)
            sg += J[k * n + i] * r[k];
        g[i] = sg;
    }
}

/* ---------- 主求解器 ---------- */

int lm_solve(double *init_params, int m, int n,
             lm_residual_t residual, lm_jacobian_t jacobian,
             void *data, const lm_config_t *config,
             lm_result_t *result)
{
    /* ---- 参数检查 ---- */
    if (n > LM_MAX_PARAMS || m > LM_MAX_RESIDUALS || m < n || !residual)
        return -2;

    lm_config_t cfg = config ? *config : default_config();
    if (cfg.max_iter <= 0) cfg.max_iter = 100;

    /* ---- 栈工作区 ---- */
    double J[LM_MAX_RESIDUALS * LM_MAX_PARAMS];
    double H[LM_MAX_PARAMS * LM_MAX_PARAMS];
    double H_aug[LM_MAX_PARAMS * LM_MAX_PARAMS];
    double L[LM_MAX_PARAMS * LM_MAX_PARAMS];
    double g[LM_MAX_PARAMS];
    double dx[LM_MAX_PARAMS];
    double x_trial[LM_MAX_PARAMS];
    double r[LM_MAX_RESIDUALS];
    double r_trial[LM_MAX_RESIDUALS];
    double tmp[LM_MAX_PARAMS];      /* λ*dx - g, 用于计算预测下降量 */

    /* ---- 初始状态 ---- */
    double *x = init_params;        /* 原地迭代 */
    residual(x, data, r, m, n);
    double cost = 0.5 * vec_dot(r, r, m);

    double lambda = cfg.lambda_init; /* < 0 表示首次迭代自动计算 */
    double nu     = 2.0;

    /* ---- LM 主循环 ---- */
    int iter = 0;
    while (iter < cfg.max_iter) {

        /* 1. Jacobian + 法方程 (x 改变后重算) */
        if (jacobian)
            jacobian(x, data, J, m, n);
        else
            numerical_jacobian(x, data, residual, m, n, J, r);

        build_normal_eqns(J, r, m, n, H, g);

        /* 2. 梯度收敛检查: ||g||_∞ < ε */
        {
            double ginf = 0.0;
            for (int i = 0; i < n; i++) {
                double ag = fabs(g[i]);
                if (ag > ginf) ginf = ag;
            }
            if (ginf < cfg.eps_grad) {
                if (result) {
                    result->iterations   = iter;
                    result->converged    = 1;
                    result->final_cost   = cost;
                    result->lambda_final = lambda;
                }
                return 0;
            }
        }

        /* 3. 首次迭代: 自动计算初始 λ */
        if (lambda < 0.0) {
            double max_diag = 0.0;
            for (int i = 0; i < n; i++) {
                double d = H[i * n + i];
                if (d > max_diag) max_diag = d;
            }
            lambda = cfg.tau * max_diag;
            if (lambda < 1e-10) lambda = 1e-10;
        }

        /* 4. 求解+增益比判断 (内层循环: 拒绝步时增大 λ 重试) */
        for (int retry = 0; retry < 30; retry++) {

            /* H_aug = H + λI, 做 Cholesky → L */
            mat_copy(H, H_aug, n, n);
            for (int i = 0; i < n; i++)
                H_aug[i * n + i] += lambda;

            if (mat_cholesky(H_aug, L, n) != 0) {
                /* 非正定 → 增大 λ */
                lambda *= nu;
                nu    *= 2.0;
                if (lambda > 1e15) { if (result) { result->iterations=iter; result->converged=0; result->final_cost=cost; result->lambda_final=lambda; } return -2; }
                continue;
            }

            /* 解 L*L^T * dx = -g */
            {
                double neg_g[LM_MAX_PARAMS];
                for (int i = 0; i < n; i++) neg_g[i] = -g[i];
                mat_cholesky_solve(L, neg_g, dx, n);
            }

            /* 试验步 */
            vec_add(x, dx, x_trial, n);
            residual(x_trial, data, r_trial, m, n);
            double cost_trial = 0.5 * vec_dot(r_trial, r_trial, m);

            /* NaN 保护 */
            if (cost_trial != cost_trial) {
                lambda *= nu; nu *= 2.0;
                if (lambda > 1e15) return -2;
                continue;
            }

            /* 增益比 ρ = 实际下降 / 预测下降
             * pred = 0.5 * dx^T * (λ*dx - g) */
            for (int i = 0; i < n; i++)
                tmp[i] = lambda * dx[i] - g[i];
            double pred = 0.5 * vec_dot(dx, tmp, n);
            if (fabs(pred) < 1e-30)
                pred = (pred >= 0) ? 1e-30 : -1e-30;

            double rho = (cost - cost_trial) / pred;

            if (rho > 0.0 || cost_trial < cost) {
                /* 接受步 */
                double cost_old = cost;
                for (int i = 0; i < n; i++) x[i] = x_trial[i];
                for (int i = 0; i < m; i++) r[i] = r_trial[i];
                cost = cost_trial;

                /* Nielsen 策略更新 λ */
                {
                    double rho_cub = 2.0 * rho - 1.0;
                    double factor = 1.0 - rho_cub * rho_cub * rho_cub;
                    if (factor < 0.333) factor = 0.333;
                    lambda *= factor;
                }
                nu = 2.0;
                iter++;

                /* 收敛检查 */
                {
                    double dx_norm = sqrt(vec_dot(dx, dx, n));
                    double x_norm  = sqrt(vec_dot(x, x, n));
                    if (dx_norm < cfg.eps_step * (x_norm + cfg.eps_step)) {
                        if (result) {
                            result->iterations   = iter;
                            result->converged    = 2;
                            result->final_cost   = cost;
                            result->lambda_final = lambda;
                        }
                        return 0;
                    }
                }
                if (fabs(cost_old - cost) < cfg.eps_cost) {
                    if (result) {
                        result->iterations   = iter;
                        result->converged    = 3;
                        result->final_cost   = cost;
                        result->lambda_final = lambda;
                    }
                    return 0;
                }

                break;  /* 退出 retry 循环, 进入下一次 LM 迭代 */
            } else {
                /* 拒绝步: 增大 λ */
                lambda *= nu;
                nu    *= 2.0;
                if (lambda > 1e15) {
                    if (result) {
                        result->iterations   = iter;
                        result->converged    = 0;
                        result->final_cost   = cost;
                        result->lambda_final = lambda;
                    }
                    return -2;
                }
            }
        }
    }

    /* 达到最大迭代次数 */
    if (result) {
        result->iterations   = iter;
        result->converged    = 0;
        result->final_cost   = cost;
        result->lambda_final = lambda;
    }
    return -1;
}
