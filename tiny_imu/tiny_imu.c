/*
 * @Author: garygoo
 * @Date: 2026-07-26
 * @Description: IMU 预积分 (Forster et al., TRO 2017)
 *   依赖: tiny_linalg (mat_mul, mat33_exp_so3, mat_copy, mat_identity,
 *         vec_cross, mat_vec_mul, mat_transpose)
 */

#include "string.h"
#include "math.h"
#include "tiny_imu.h"
#include "../tiny_linalg/tiny_linalg.h"

/* ---------- 静态辅助函数 ---------- */

/* 3-向量 → 3×3 反对称矩阵 (skew-symmetric) */
static void skew_mat(const double v[3], double S[9])
{
    S[0] =  0.0;    S[1] = -v[2];   S[2] =  v[1];
    S[3] =  v[2];   S[4] =  0.0;    S[5] = -v[0];
    S[6] = -v[1];   S[7] =  v[0];   S[8] =  0.0;
}

/* ---------- 公开 API ---------- */

void imu_preint_init(imu_preint_t *p,
                     double sigma_a, double sigma_g,
                     double sigma_ba, double sigma_bg,
                     const double ba[3], const double bg[3])
{
    mat_identity(p->dR, 3);
    for (int i = 0; i < 3; i++) {
        p->dv[i] = 0.0;
        p->dp[i] = 0.0;
    }
    p->dt = 0.0;

    for (int i = 0; i < 81; i++) p->cov[i] = 0.0;

    for (int i = 0; i < 9; i++) {
        p->J_dR_bg[i] = 0.0;
        p->J_dv_ba[i] = 0.0;
        p->J_dv_bg[i] = 0.0;
        p->J_dp_ba[i] = 0.0;
        p->J_dp_bg[i] = 0.0;
    }

    p->sigma_a  = sigma_a;
    p->sigma_g  = sigma_g;
    p->sigma_ba = sigma_ba;
    p->sigma_bg = sigma_bg;

    for (int i = 0; i < 3; i++) {
        p->ba[i] = ba[i];
        p->bg[i] = bg[i];
    }
}

/* 核心: 一次 IMU 采样的预积分 + 误差状态协方差/Jacobian 传播
 *
 * 误差状态 η = [δθ(3), δv(3), δp(3)]^T 的离散时间传播:
 *   η_{k+1} = A · η_k + B · n_meas
 *   Σ_{k+1} = A · Σ_k · A^T + B · Q_meas · B^T
 *
 * 偏置 Jacobian 传播 (每个 3×3):
 *   J_bg = [J_dR_bg; J_dv_bg; J_dp_bg]   (9×3)
 *   J_ba = [0; J_dv_ba; J_dp_ba]         (9×3, 第一块永为零)
 *   J_{k+1} = A · J_k + B_bias·dt */
void imu_preint_update(imu_preint_t *p,
                       const double acc[3], const double gyr[3],
                       double dt)
{
    if (dt <= 0.0) return;

    /* 保存更新前的值 (Jacobian 传播需要旧值) */
    double dR_old[9], dv_old[3];
    mat_copy(p->dR, dR_old, 3, 3);
    for (int i = 0; i < 3; i++) dv_old[i] = p->dv[i];

    double J_dR_bg_old[9], J_dv_ba_old[9], J_dv_bg_old[9];
    double J_dp_ba_old[9], J_dp_bg_old[9];
    mat_copy(p->J_dR_bg, J_dR_bg_old, 3, 3);
    mat_copy(p->J_dv_ba, J_dv_ba_old, 3, 3);
    mat_copy(p->J_dv_bg, J_dv_bg_old, 3, 3);
    mat_copy(p->J_dp_ba, J_dp_ba_old, 3, 3);
    mat_copy(p->J_dp_bg, J_dp_bg_old, 3, 3);

    /* ---- 1. 校正测量值, 计算导出量 ---- */
    double w[3], a[3];
    for (int i = 0; i < 3; i++) {
        w[i] = gyr[i] - p->bg[i];
        a[i] = acc[i] - p->ba[i];
    }

    double w_dt[3], a_dt[3];
    for (int i = 0; i < 3; i++) {
        w_dt[i] = w[i] * dt;
        a_dt[i] = a[i] * dt;
    }
    double dt2 = 0.5 * dt * dt;

    /* ---- 2. 名义状态更新 (前向 Euler) ---- */

    /* ΔR += Exp(w·dt)  (右乘) */
    double dR_inc[9];
    mat33_exp_so3(w_dt, dR_inc);
    mat_mul(dR_old, dR_inc, p->dR, 3, 3, 3);

    /* Δv += R_old · a · dt */
    double dv_inc_3[3];
    mat_vec_mul(dR_old, a_dt, dv_inc_3, 3, 3);
    for (int i = 0; i < 3; i++) p->dv[i] += dv_inc_3[i];

    /* Δp += v_old·dt + 0.5·R_old·a·dt² */
    double a_dt2[3], dp_rot[3];
    for (int i = 0; i < 3; i++) a_dt2[i] = a[i] * dt2;
    mat_vec_mul(dR_old, a_dt2, dp_rot, 3, 3);
    for (int i = 0; i < 3; i++)
        p->dp[i] += dv_old[i] * dt + dp_rot[i];

    p->dt += dt;

    /* ---- 3. 误差状态转移矩阵 A (9×9) ---- */

    /* A00 = Exp(-w·dt) = Exp(w·dt)^T, 3×3 */
    double w_neg[3], A00[9];
    for (int i = 0; i < 3; i++) w_neg[i] = -w_dt[i];
    mat33_exp_so3(w_neg, A00);

    /* A10 = -R_old · skew(a·dt) */
    double S_a_dt[9], A10[9];
    skew_mat(a_dt, S_a_dt);
    for (int i = 0; i < 9; i++) S_a_dt[i] = -S_a_dt[i];
    mat_mul(dR_old, S_a_dt, A10, 3, 3, 3);

    /* A20 = -R_old · skew(a·dt²/2) */
    double S_a_dt2[9], A20[9];
    skew_mat(a_dt2, S_a_dt2);
    for (int i = 0; i < 9; i++) S_a_dt2[i] = -S_a_dt2[i];
    mat_mul(dR_old, S_a_dt2, A20, 3, 3, 3);

    /* 组装 A = [A00, 0, 0; A10, I, 0; A20, I·dt, I] */
    double A[81];
    for (int i = 0; i < 81; i++) A[i] = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[(0+i)*9 + (0+j)] = A00[i*3 + j];
            A[(3+i)*9 + (0+j)] = A10[i*3 + j];
            A[(6+i)*9 + (0+j)] = A20[i*3 + j];
        }
        A[(3+i)*9 + (3+i)] = 1.0;
        A[(6+i)*9 + (3+i)] = dt;
        A[(6+i)*9 + (6+i)] = 1.0;
    }

    /* ---- 4. 噪声输入 B·Q·B^T (9×9) ---- */

    /* Q_meas = diag(σ_g²·I, σ_a²·I), 6×6 */
    double sg2 = p->sigma_g * p->sigma_g;
    double sa2 = p->sigma_a * p->sigma_a;

    /* 直接计算 B·Q·B^T 的 9×9 结果, 利用 B 的块结构避免显式构造 B:
     * BQBt[0:3, 0:3] = sg2·dt²·I
     * BQBt[3:6, 3:6] = sa2·dt²·R_old·R_old^T = sa2·dt²·I  (R_old 是旋转矩阵)
     * BQBt[3:6, 0:3] = 0  (不同噪声源, 不相关)
     * BQBt[6:9, 3:6] = sa2·dt²·dt2·R_old^T = sa2·dt·dt2·R_old^T...
     * 算了, 显式构造 B 更清晰不容易错。 */
    double B[54];  /* 9×6 */
    for (int i = 0; i < 54; i++) B[i] = 0.0;

    /* B[0:3, 0:3] = I·dt */
    for (int i = 0; i < 3; i++) B[i*6 + i] = dt;

    /* B[3:6, 3:6] = R_old·dt */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[(3+i)*6 + (3+j)] = dR_old[i*3 + j] * dt;

    /* B[6:9, 3:6] = R_old·dt²/2 */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[(6+i)*6 + (3+j)] = dR_old[i*3 + j] * dt2;

    double Q_meas[36];
    for (int i = 0; i < 36; i++) Q_meas[i] = 0.0;
    for (int i = 0; i < 3; i++) {
        Q_meas[i*6 + i]         = sg2;
        Q_meas[(3+i)*6 + (3+i)] = sa2;
    }

    double BQ[54], Bt[54], BQBt[81];
    mat_mul(B, Q_meas, BQ, 9, 6, 6);
    mat_transpose(B, Bt, 9, 6);
    mat_mul(BQ, Bt, BQBt, 9, 6, 9);

    /* ---- 5. 协方差更新: cov = A·cov·A^T + BQBt ---- */
    double temp[81], At[81];
    mat_mul(A, p->cov, temp, 9, 9, 9);
    mat_transpose(A, At, 9, 9);
    mat_mul(temp, At, p->cov, 9, 9, 9);
    for (int i = 0; i < 81; i++) p->cov[i] += BQBt[i];

    /* ---- 6. 偏置 Jacobian 传播 ----
     * 传播公式:
     *   J_dR_bg_new = A00 · J_dR_bg_old - Jr(w·dt) · dt
     *   J_dv_bg_new = A10 · J_dR_bg_old + J_dv_bg_old
     *   J_dp_bg_new = A20 · J_dR_bg_old + J_dv_bg_old·dt + J_dp_bg_old
     *   J_dv_ba_new = J_dv_ba_old - R_old · dt
     *   J_dp_ba_new = J_dp_ba_old + J_dv_ba_old·dt - 0.5·R_old·dt²
     * 注意: 必须全部使用旧 Jacobian 值 */

    double Jr[9];
    mat33_right_jacobian(w_dt, Jr);

    /* J_dR_bg */
    {
        double tmp[9];
        mat_mul(A00, J_dR_bg_old, tmp, 3, 3, 3);
        for (int i = 0; i < 9; i++)
            p->J_dR_bg[i] = tmp[i] - Jr[i] * dt;
    }

    /* J_dv_bg = A10 · J_dR_bg_old + J_dv_bg_old */
    {
        double A10_Jr[9];
        mat_mul(A10, J_dR_bg_old, A10_Jr, 3, 3, 3);
        for (int i = 0; i < 9; i++)
            p->J_dv_bg[i] = A10_Jr[i] + J_dv_bg_old[i];
    }

    /* J_dp_bg = A20 · J_dR_bg_old + J_dv_bg_old·dt + J_dp_bg_old */
    {
        double A20_Jr[9];
        mat_mul(A20, J_dR_bg_old, A20_Jr, 3, 3, 3);
        for (int i = 0; i < 9; i++)
            p->J_dp_bg[i] = A20_Jr[i] + J_dv_bg_old[i]*dt + J_dp_bg_old[i];
    }

    /* J_dv_ba = J_dv_ba_old - R_old · dt */
    {
        for (int i = 0; i < 9; i++)
            p->J_dv_ba[i] = J_dv_ba_old[i] - dR_old[i] * dt;
    }

    /* J_dp_ba = J_dp_ba_old + J_dv_ba_old·dt - R_old·dt²/2 */
    {
        for (int i = 0; i < 9; i++)
            p->J_dp_ba[i] = J_dp_ba_old[i] + J_dv_ba_old[i]*dt - dR_old[i]*dt2;
    }
}

/* ---------- 重置 ---------- */

void imu_preint_reset(imu_preint_t *p,
                      const double ba[3], const double bg[3])
{
    mat_identity(p->dR, 3);
    for (int i = 0; i < 3; i++) {
        p->dv[i] = 0.0;
        p->dp[i] = 0.0;
    }
    p->dt = 0.0;

    for (int i = 0; i < 81; i++) p->cov[i] = 0.0;

    for (int i = 0; i < 9; i++) {
        p->J_dR_bg[i] = 0.0;
        p->J_dv_ba[i] = 0.0;
        p->J_dv_bg[i] = 0.0;
        p->J_dp_ba[i] = 0.0;
        p->J_dp_bg[i] = 0.0;
    }

    for (int i = 0; i < 3; i++) {
        p->ba[i] = ba[i];
        p->bg[i] = bg[i];
    }
}

/* ---------- 提取预积分量 ---------- */

void imu_preint_get_delta(const imu_preint_t *p,
                          double dR[9], double dv[3], double dp[3])
{
    mat_copy(p->dR, dR, 3, 3);
    for (int i = 0; i < 3; i++) {
        dv[i] = p->dv[i];
        dp[i] = p->dp[i];
    }
}

/* ---------- 一阶偏置修正 ---------- */

void imu_preint_bias_correct(imu_preint_t *p,
                             const double ba_new[3],
                             const double bg_new[3])
{
    double dba[3], dbg[3];
    for (int i = 0; i < 3; i++) {
        dba[i] = ba_new[i] - p->ba[i];
        dbg[i] = bg_new[i] - p->bg[i];
    }

    /* ΔR(bg_new) ≈ ΔR(bg) · Exp(J_dR_bg · δbg) */
    double dR_bg[3], dR_corr[9];
    mat_vec_mul(p->J_dR_bg, dbg, dR_bg, 3, 3);
    mat33_exp_so3(dR_bg, dR_corr);
    {
        double dR_tmp[9];
        mat_mul(p->dR, dR_corr, dR_tmp, 3, 3, 3);
        mat_copy(dR_tmp, p->dR, 3, 3);
    }

    /* Δv += J_dv_ba·δba + J_dv_bg·δbg */
    {
        double dv_ba[3], dv_bg[3];
        mat_vec_mul(p->J_dv_ba, dba, dv_ba, 3, 3);
        mat_vec_mul(p->J_dv_bg, dbg, dv_bg, 3, 3);
        for (int i = 0; i < 3; i++)
            p->dv[i] += dv_ba[i] + dv_bg[i];
    }

    /* Δp += J_dp_ba·δba + J_dp_bg·δbg */
    {
        double dp_ba[3], dp_bg[3];
        mat_vec_mul(p->J_dp_ba, dba, dp_ba, 3, 3);
        mat_vec_mul(p->J_dp_bg, dbg, dp_bg, 3, 3);
        for (int i = 0; i < 3; i++)
            p->dp[i] += dp_ba[i] + dp_bg[i];
    }

    /* 更新偏置引用 */
    for (int i = 0; i < 3; i++) {
        p->ba[i] = ba_new[i];
        p->bg[i] = bg_new[i];
    }
}
