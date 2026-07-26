#ifndef TINY_OPT_H
#define TINY_OPT_H

/* ============================================================
 * tiny_opt — Levenberg-Marquardt 非线性优化求解器
 *
 * 纯 C99, 零依赖 (仅需 tiny_linalg), 栈分配, 无 malloc。
 * 目标: MaixCAM-Pro (RISC-V 64, musl libc) 上的 PnP/EKF 优化后端。
 * ============================================================ */

/* 最大问题规模 (栈分配上限) */
#define LM_MAX_PARAMS     48
#define LM_MAX_RESIDUALS  160

/* ---------- 回调类型 ---------- */

/* 残差函数: params[n] → residual[m]
 * data: 用户自定义问题数据 (观测点、图像坐标等)
 * 调用方保证 params 在函数返回前不被修改 */
typedef void (*lm_residual_t)(const double *params, void *data,
                               double *residual, int m, int n);

/* Jacobian 函数: params[n] → J[m×n] (行优先)
 * 若传 NULL, lm_solve 内部用前向差分数值近似 */
typedef void (*lm_jacobian_t)(const double *params, void *data,
                               double *J, int m, int n);

/* ---------- 配置 ---------- */

typedef struct {
    int    max_iter;       /* 最大迭代次数 (默认 100)                */
    double tau;            /* 初始 λ 缩放因子 (默认 1e-3)           */
    double eps_grad;       /* 梯度无穷范数收敛阈值 (默认 1e-8)       */
    double eps_step;       /* 步长收敛阈值 (默认 1e-8)              */
    double eps_cost;       /* 代价变化收敛阈值 (默认 1e-8)           */
    double lambda_init;    /* 初始 λ; 负值=自动 tau*max(diag(H0))  */
} lm_config_t;

/* ---------- 结果 ---------- */

typedef struct {
    int    iterations;     /* 实际迭代次数                          */
    int    converged;      /* 0=未收敛 1=梯度 2=步长 3=代价变化      */
    double final_cost;     /* 最终代价 0.5 * ||r||^2                 */
    double lambda_final;   /* 最终 λ                                */
} lm_result_t;

/* ---------- 主求解器 ---------- */

/* 返回值: 0=收敛, -1=达到最大迭代, -2=λ过大无法继续
 *
 * 参数:
 *   init_params  - 初值 n 维, 会在此数组内原地迭代
 *   m, n         - 残差维度, 参数维度 (m≥n, n≤LM_MAX_PARAMS, m≤LM_MAX_RESIDUALS)
 *   residual     - 残差回调
 *   jacobian     - Jacobian 回调 (可为 NULL, 表示数值微分)
 *   data         - 透传给回调的用户数据
 *   config       - 配置 (可为 NULL, 表示全部默认)
 *   result       - 输出结果 (可为 NULL)
 */
int lm_solve(double *init_params, int m, int n,
             lm_residual_t residual, lm_jacobian_t jacobian,
             void *data, const lm_config_t *config,
             lm_result_t *result);

#endif /* TINY_OPT_H */
