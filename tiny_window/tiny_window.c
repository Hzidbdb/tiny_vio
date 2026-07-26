/* ============================================================
 * tiny_window.c — 滑动窗口紧耦合优化器实现
 *
 * 依赖: tiny_window.h, tiny_imu.h, tiny_linalg.h, tiny_opt.h
 * ============================================================ */

#include "stdio.h"
#include "math.h"
#include "tiny_window.h"
#include "../tiny_linalg/tiny_linalg.h"
#include "../tiny_opt/tiny_opt.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- 窗口: 下标辅助 ---------- */

/* 获取第 i 个关键帧在 kf[] 中的实际下标 (0=最老, n_kf-1=最新) */
static int idx(const window_t *w, int i)
{
    return (w->start + i) % WINDOW_MAX_KF;
}

/* 状态向量的偏移量:
 * v_0        → 0
 * ξ_i (i≥1)  → 3 + (i-1)*9
 * v_i (i≥1)  → 3 + (i-1)*9 + 6
 * ba         → 3 + (n_kf-1)*9
 * bg         → 3 + (n_kf-1)*9 + 3          */
static int off_v(int i)       { return (i == 0) ? 0 : (3 + (i-1)*9 + 6); }
static int off_xi(int i)      { return 3 + (i-1)*9; }       /* i ≥ 1 */
static int off_ba(int n_kf)   { return 3 + (n_kf-1)*9; }
static int off_bg(int n_kf)   { return 3 + (n_kf-1)*9 + 3; }

/* 将一个 se(3) 6 维向量展开: R = Exp(ξ[0..2]), t = ξ[3..5] */
static void se3_exp(const double xi[6], double R[9], double t[3])
{
    mat33_exp_so3(xi, R);   /* rotation from first 3 elements */
    t[0] = xi[3]; t[1] = xi[4]; t[2] = xi[5];
}

/* 将 R,t 压缩为 se(3): ξ[0..2] = log(R), ξ[3..5] = t */
static void se3_log(const double R[9], const double t[3], double xi[6])
{
    mat33_log_so3(R, xi);
    xi[3] = t[0]; xi[4] = t[1]; xi[5] = t[2];
}

/* ---------- 辅助: 3×3 反对称矩阵 ---------- */

static void skew3(const double v[3], double M[9])
{
    M[0]=0;    M[1]=-v[2]; M[2]=v[1];
    M[3]=v[2]; M[4]=0;     M[5]=-v[0];
    M[6]=-v[1];M[7]=v[0];  M[8]=0;
}

/* 将 3×3 块填到 J (m×n 行优先) 的第 (row, col) 处 */
static void fill_J_block(double *J, int m, int n, int row, int col,
                         const double block[9])
{
    (void)m;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            J[(row + r) * n + (col + c)] = block[r * 3 + c];
}

/* 将 3×3 块的各行乘以标量权重 w[3] */
static void apply_row_weights(double block[9], const double w[3])
{
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            block[r * 3 + c] *= w[r];
}

/* ---------- 残差计算 ---------- */

typedef struct {
    const window_t *w;
    int    n_state;     /* 当前实际状态维度 */
    int    n_resid;     /* 当前实际残差维度 */
    int    n_kf;        /* 快照: 优化开始时的关键帧数 */
    int    ba_off;      /* ba 在状态向量中的偏移 (固定, 从 n_kf 预计算) */
    int    bg_off;      /* bg 在状态向量中的偏移 */
    double ba_lin[3];   /* 线性化点偏置 (存储的预积分偏置) */
    double bg_lin[3];
    double huber_wt[WINDOW_MAX_RESID];  /* Huber IRLS 权重 (残差→Jacobi 传递) */
} resid_ctx_t;

static void window_residual(const double *x, void *data,
                            double *r, int m, int n)
{
    resid_ctx_t *ctx = (resid_ctx_t *)data;
    const window_t *w = ctx->w;
    int n_kf = ctx->n_kf;
    (void)m; (void)n;

    /* 提取偏置 */
    const double *ba_cur = x + ctx->ba_off;
    const double *bg_cur = x + ctx->bg_off;

    int ri = 0;  /* 残差下标 */

    for (int i = 0; i < n_kf - 1; i++) {
        int i0 = idx(w, i), i1 = idx(w, i + 1);
        const window_kf_t *kf0 = &w->kf[i0];
        const window_kf_t *kf1 = &w->kf[i1];

        double dt = kf1->preint.dt;

        /* --- 提取状态 --- */
        double R_i[9], t_i[3], v_i[3];
        if (i == 0) {
            /* kf_0 位姿固定, 速度用 x 中的估计值 */
            mat_copy(kf0->R, R_i, 3, 3);
            mat_copy(kf0->t, t_i, 3, 1);
            v_i[0] = x[0]; v_i[1] = x[1]; v_i[2] = x[2];
        } else {
            se3_exp(x + off_xi(i), R_i, t_i);
            const double *xv = x + off_v(i);
            v_i[0] = xv[0]; v_i[1] = xv[1]; v_i[2] = xv[2];
        }

        double R_ip1[9], t_ip1[3], v_ip1[3];
        se3_exp(x + off_xi(i + 1), R_ip1, t_ip1);
        const double *xv1 = x + off_v(i + 1);
        v_ip1[0] = xv1[0]; v_ip1[1] = xv1[1]; v_ip1[2] = xv1[2];

        /* --- 偏置修正预积分 --- */
        double dba[3] = {ba_cur[0]-ctx->ba_lin[0],
                         ba_cur[1]-ctx->ba_lin[1],
                         ba_cur[2]-ctx->ba_lin[2]};
        double dbg[3] = {bg_cur[0]-ctx->bg_lin[0],
                         bg_cur[1]-ctx->bg_lin[1],
                         bg_cur[2]-ctx->bg_lin[2]};

        /* dR_corr = dR * Exp(J_dR_bg * δbg) */
        double dR_bg[3], dR_corr[9];
        mat_vec_mul(kf1->preint.J_dR_bg, dbg, dR_bg, 3, 3);
        double dR_exp[9];
        mat33_exp_so3(dR_bg, dR_exp);
        mat_mul(kf1->preint.dR, dR_exp, dR_corr, 3, 3, 3);

        /* dv_corr = dv + J_dv_ba·δba + J_dv_bg·δbg */
        double dv_ba[3], dv_bg[3], dv_corr[3];
        mat_vec_mul(kf1->preint.J_dv_ba, dba, dv_ba, 3, 3);
        mat_vec_mul(kf1->preint.J_dv_bg, dbg, dv_bg, 3, 3);
        dv_corr[0] = kf1->preint.dv[0] + dv_ba[0] + dv_bg[0];
        dv_corr[1] = kf1->preint.dv[1] + dv_ba[1] + dv_bg[1];
        dv_corr[2] = kf1->preint.dv[2] + dv_ba[2] + dv_bg[2];

        /* dp_corr = dp + J_dp_ba·δba + J_dp_bg·δbg */
        double dp_ba[3], dp_bg[3], dp_corr[3];
        mat_vec_mul(kf1->preint.J_dp_ba, dba, dp_ba, 3, 3);
        mat_vec_mul(kf1->preint.J_dp_bg, dbg, dp_bg, 3, 3);
        dp_corr[0] = kf1->preint.dp[0] + dp_ba[0] + dp_bg[0];
        dp_corr[1] = kf1->preint.dp[1] + dp_ba[1] + dp_bg[1];
        dp_corr[2] = kf1->preint.dp[2] + dp_ba[2] + dp_bg[2];

        /* --- IMU 旋转残差: r_R = log(dR_corr^T * R_i^T * R_{i+1}) --- */
        double RT_i[9], R_rel[9];
        mat_transpose(R_i, RT_i, 3, 3);
        double tmp1[9];
        mat_mul(RT_i, R_ip1, tmp1, 3, 3, 3);
        double dRT[9];
        mat_transpose(dR_corr, dRT, 3, 3);
        mat_mul(dRT, tmp1, R_rel, 3, 3, 3);
        mat33_log_so3(R_rel, r + ri);

        /* --- IMU 速度残差: r_v = R_i^T*(v_{i+1}-v_i-g*dt) - dv_corr --- */
        double dv_world[3] = {v_ip1[0]-v_i[0] - w->g[0]*dt,
                              v_ip1[1]-v_i[1] - w->g[1]*dt,
                              v_ip1[2]-v_i[2] - w->g[2]*dt};
        double dv_body[3];
        mat_vec_mul(RT_i, dv_world, dv_body, 3, 3);
        r[ri+3] = dv_body[0] - dv_corr[0];
        r[ri+4] = dv_body[1] - dv_corr[1];
        r[ri+5] = dv_body[2] - dv_corr[2];

        /* --- IMU 位置残差: r_p = R_i^T*(t_{i+1}-t_i-v_i*dt-½g*dt²) - dp_corr --- */
        double half_dt2 = 0.5 * dt * dt;
        double dp_world[3] = {t_ip1[0]-t_i[0] - v_i[0]*dt - w->g[0]*half_dt2,
                              t_ip1[1]-t_i[1] - v_i[1]*dt - w->g[1]*half_dt2,
                              t_ip1[2]-t_i[2] - v_i[2]*dt - w->g[2]*half_dt2};
        double dp_body[3];
        mat_vec_mul(RT_i, dp_world, dp_body, 3, 3);
        r[ri+6] = dp_body[0] - dp_corr[0];
        r[ri+7] = dp_body[1] - dp_corr[1];
        r[ri+8] = dp_body[2] - dp_corr[2];

        /* --- 协方差白化: 用对角线权重 --- */
        const double *cov = kf1->preint.cov;
        double w_rx = (cov[0] > 1e-30) ? 1.0/sqrt(cov[0]) : 1e6;
        double w_ry = (cov[10] > 1e-30) ? 1.0/sqrt(cov[10]) : 1e6;
        double w_rz = (cov[20] > 1e-30) ? 1.0/sqrt(cov[20]) : 1e6;
        double w_vx = (cov[30] > 1e-30) ? 1.0/sqrt(cov[30]) : 1e6;
        double w_vy = (cov[40] > 1e-30) ? 1.0/sqrt(cov[40]) : 1e6;
        double w_vz = (cov[50] > 1e-30) ? 1.0/sqrt(cov[50]) : 1e6;
        double w_px = (cov[60] > 1e-30) ? 1.0/sqrt(cov[60]) : 1e6;
        double w_py = (cov[70] > 1e-30) ? 1.0/sqrt(cov[70]) : 1e6;
        double w_pz = (cov[80] > 1e-30) ? 1.0/sqrt(cov[80]) : 1e6;

        r[ri+0] *= w_rx;  r[ri+1] *= w_ry;  r[ri+2] *= w_rz;
        r[ri+3] *= w_vx;  r[ri+4] *= w_vy;  r[ri+5] *= w_vz;
        r[ri+6] *= w_px;  r[ri+7] *= w_py;  r[ri+8] *= w_pz;
        ri += 9;
    }

    /* --- PnP 视觉残差 --- */
    for (int i = 1; i < n_kf; i++) {
        int ik = idx(w, i);
        const window_kf_t *kf = &w->kf[ik];

        double R_i[9], t_i[3];
        se3_exp(x + off_xi(i), R_i, t_i);

        /* r_vis_R = log(R_i^T * R_pnp) */
        double RT_i[9], R_err[9];
        mat_transpose(R_i, RT_i, 3, 3);
        mat_mul(RT_i, kf->R_meas, R_err, 3, 3, 3);
        mat33_log_so3(R_err, r + ri);

        /* r_vis_t = t_i - t_pnp */
        r[ri+3] = t_i[0] - kf->t_meas[0];
        r[ri+4] = t_i[1] - kf->t_meas[1];
        r[ri+5] = t_i[2] - kf->t_meas[2];

        /* 视觉权重 */
        double wr = 1.0 / w->sigma_vis_r;
        double wt = 1.0 / w->sigma_vis_t;
        r[ri+0] *= wr; r[ri+1] *= wr; r[ri+2] *= wr;
        r[ri+3] *= wt; r[ri+4] *= wt; r[ri+5] *= wt;
        ri += 6;
    }

    /* v_0 速度先验: r = v_0 / σ, 约束初始速度接近 0 */
    if (w->sigma_v0_prior > 0) {
        double wr0 = 1.0 / w->sigma_v0_prior;
        r[ri+0] = x[0] * wr0;
        r[ri+1] = x[1] * wr0;
        r[ri+2] = x[2] * wr0;
        ri += 3;
    }

    /* Huber kernel: IRLS weighting, √w = 1 if |r| ≤ δ, else √(δ/|r|) */
    if (w->huber_delta > 0) {
        for (int k = 0; k < m; k++) {
            double abs_r = fabs(r[k]);
            if (abs_r > w->huber_delta) {
                double wt = sqrt(w->huber_delta / abs_r);
                ctx->huber_wt[k] = wt;
                r[k] *= wt;
            } else {
                ctx->huber_wt[k] = 1.0;
            }
        }
    } else {
        for (int k = 0; k < m; k++) ctx->huber_wt[k] = 1.0;
    }
}

/* ---------- 解析 Jacobian ---------- */

static void window_jacobian(const double *x, void *data,
                            double *J, int m, int n)
{
    resid_ctx_t *ctx = (resid_ctx_t *)data;
    const window_t *w = ctx->w;
    int n_kf = ctx->n_kf;

    /* 清零 J */
    for (int i = 0; i < m * n; i++) J[i] = 0.0;

    const double *bg_cur = x + ctx->bg_off;

    /* 预提取所有关键帧状态 */
    double R_kf[WINDOW_MAX_KF][9], t_kf[WINDOW_MAX_KF][3], v_kf[WINDOW_MAX_KF][3];
    double Jr_kf[WINDOW_MAX_KF][9];  /* Jr(log(R_i)), 旋转参数化转换用 */

    int i0 = idx(w, 0);
    mat_copy(w->kf[i0].R, R_kf[0], 3, 3);
    mat_copy(w->kf[i0].t, t_kf[0], 3, 1);
    v_kf[0][0] = x[0]; v_kf[0][1] = x[1]; v_kf[0][2] = x[2];
    {
        double phi0[3];
        mat33_log_so3(R_kf[0], phi0);
        mat33_right_jacobian(phi0, Jr_kf[0]);
    }

    for (int i = 1; i < n_kf; i++) {
        se3_exp(x + off_xi(i), R_kf[i], t_kf[i]);
        const double *xv = x + off_v(i);
        v_kf[i][0] = xv[0]; v_kf[i][1] = xv[1]; v_kf[i][2] = xv[2];

        double phi_i[3];
        mat_copy(x + off_xi(i), phi_i, 3, 1);
        mat33_right_jacobian(phi_i, Jr_kf[i]);
    }

    double dbg[3] = {bg_cur[0]-ctx->bg_lin[0],
                     bg_cur[1]-ctx->bg_lin[1],
                     bg_cur[2]-ctx->bg_lin[2]};

    int ri = 0;

    /* ===== IMU 因子 (i → i+1) ===== */
    for (int i = 0; i < n_kf - 1; i++) {
        int i1 = idx(w, i + 1);
        const window_kf_t *kf1 = &w->kf[i1];
        double dt = kf1->preint.dt;

        /* 偏置修正预积分 */
        double dR_bg[3], dR_corr[9];
        mat_vec_mul(kf1->preint.J_dR_bg, dbg, dR_bg, 3, 3);
        double dR_exp[9];
        mat33_exp_so3(dR_bg, dR_exp);
        mat_mul(kf1->preint.dR, dR_exp, dR_corr, 3, 3, 3);

        /* 公共中间量 */
        double RT_i[9];
        mat_transpose(R_kf[i], RT_i, 3, 3);

        /* 原始旋转残差 + Jr^{-1}(r_R) */
        double tmp1[9], R_rel[9];
        mat_mul(RT_i, R_kf[i+1], tmp1, 3, 3, 3);
        double dRT[9];
        mat_transpose(dR_corr, dRT, 3, 3);
        mat_mul(dRT, tmp1, R_rel, 3, 3, 3);
        double r_R[3];
        mat33_log_so3(R_rel, r_R);
        double Jr_inv_R[9];
        mat33_right_jacobian_inv(r_R, Jr_inv_R);

        /* 速度/位置残差中间量 */
        double dv_world[3] = {v_kf[i+1][0]-v_kf[i][0] - w->g[0]*dt,
                              v_kf[i+1][1]-v_kf[i][1] - w->g[1]*dt,
                              v_kf[i+1][2]-v_kf[i][2] - w->g[2]*dt};
        double dv_body[3], dv_skew[9];
        mat_vec_mul(RT_i, dv_world, dv_body, 3, 3);
        skew3(dv_body, dv_skew);

        double half_dt2 = 0.5 * dt * dt;
        double dp_world[3] = {t_kf[i+1][0]-t_kf[i][0] - v_kf[i][0]*dt - w->g[0]*half_dt2,
                              t_kf[i+1][1]-t_kf[i][1] - v_kf[i][1]*dt - w->g[1]*half_dt2,
                              t_kf[i+1][2]-t_kf[i][2] - v_kf[i][2]*dt - w->g[2]*half_dt2};
        double dp_body[3], dp_skew[9];
        mat_vec_mul(RT_i, dp_world, dp_body, 3, 3);
        skew3(dp_body, dp_skew);

        /* 协方差白化权重 */
        const double *cov = kf1->preint.cov;
        double wr[3] = {
            (cov[0] > 1e-30) ? 1.0/sqrt(cov[0]) : 1e6,
            (cov[10] > 1e-30) ? 1.0/sqrt(cov[10]) : 1e6,
            (cov[20] > 1e-30) ? 1.0/sqrt(cov[20]) : 1e6
        };
        double wv[3] = {
            (cov[30] > 1e-30) ? 1.0/sqrt(cov[30]) : 1e6,
            (cov[40] > 1e-30) ? 1.0/sqrt(cov[40]) : 1e6,
            (cov[50] > 1e-30) ? 1.0/sqrt(cov[50]) : 1e6
        };
        double wp[3] = {
            (cov[60] > 1e-30) ? 1.0/sqrt(cov[60]) : 1e6,
            (cov[70] > 1e-30) ? 1.0/sqrt(cov[70]) : 1e6,
            (cov[80] > 1e-30) ? 1.0/sqrt(cov[80]) : 1e6
        };

        double block[9], tmp[9];
        int roff = ri;

        /* --- r_R: rows roff..roff+2 --- */

        /* ∂r_R/∂φ_i = -Jr^{-1}(r_R)·dR_corr^T·Jr(φ_i)  [i≥1] */
        if (i >= 1) {
            mat_mul(Jr_inv_R, dRT, block, 3, 3, 3);        /* Jr^{-1} * dR_corr^T */
            for (int k = 0; k < 9; k++) block[k] = -block[k];
            mat_mul(block, Jr_kf[i], tmp, 3, 3, 3);        /* × Jr(φ_i) */
            apply_row_weights(tmp, wr);
            fill_J_block(J, m, n, roff, off_xi(i), tmp);
        }

        /* ∂r_R/∂φ_{i+1} = Jr^{-1}(r_R)·Jr(φ_{i+1}) */
        mat_mul(Jr_inv_R, Jr_kf[i+1], block, 3, 3, 3);
        apply_row_weights(block, wr);
        fill_J_block(J, m, n, roff, off_xi(i+1), block);

        /* ∂r_R/∂bg = -Jr^{-1}(r_R)·J_dR_bg */
        mat_mul(Jr_inv_R, kf1->preint.J_dR_bg, block, 3, 3, 3);
        for (int k = 0; k < 9; k++) block[k] = -block[k];
        apply_row_weights(block, wr);
        fill_J_block(J, m, n, roff, ctx->bg_off, block);

        /* --- r_v: rows roff+3..roff+5 --- */
        roff += 3;

        /* ∂r_v/∂φ_i = skew(R_i^T·Δv_world)·Jr(φ_i)  [i≥1] */
        if (i >= 1) {
            mat_mul(dv_skew, Jr_kf[i], block, 3, 3, 3);
            apply_row_weights(block, wv);
            fill_J_block(J, m, n, roff, off_xi(i), block);
        }

        /* ∂r_v/∂v_i = -R_i^T */
        for (int k = 0; k < 9; k++) block[k] = -RT_i[k];
        apply_row_weights(block, wv);
        fill_J_block(J, m, n, roff, (i == 0) ? 0 : off_v(i), block);

        /* ∂r_v/∂v_{i+1} = R_i^T */
        mat_copy(RT_i, block, 3, 3);
        apply_row_weights(block, wv);
        fill_J_block(J, m, n, roff, off_v(i+1), block);

        /* ∂r_v/∂ba = -J_dv_ba */
        for (int k = 0; k < 9; k++) block[k] = -kf1->preint.J_dv_ba[k];
        apply_row_weights(block, wv);
        fill_J_block(J, m, n, roff, ctx->ba_off, block);

        /* ∂r_v/∂bg = -J_dv_bg */
        for (int k = 0; k < 9; k++) block[k] = -kf1->preint.J_dv_bg[k];
        apply_row_weights(block, wv);
        fill_J_block(J, m, n, roff, ctx->bg_off, block);

        /* --- r_p: rows roff+3..roff+5 --- */
        roff += 3;

        /* ∂r_p/∂φ_i = skew(R_i^T·Δp_world)·Jr(φ_i)  [i≥1] */
        if (i >= 1) {
            mat_mul(dp_skew, Jr_kf[i], block, 3, 3, 3);
            apply_row_weights(block, wp);
            fill_J_block(J, m, n, roff, off_xi(i), block);

            /* ∂r_p/∂t_i = -R_i^T */
            for (int k = 0; k < 9; k++) block[k] = -RT_i[k];
            apply_row_weights(block, wp);
            fill_J_block(J, m, n, roff, off_xi(i) + 3, block);
        }

        /* ∂r_p/∂v_i = -R_i^T * dt */
        for (int k = 0; k < 9; k++) block[k] = -RT_i[k] * dt;
        apply_row_weights(block, wp);
        fill_J_block(J, m, n, roff, (i == 0) ? 0 : off_v(i), block);

        /* ∂r_p/∂t_{i+1} = R_i^T */
        mat_copy(RT_i, block, 3, 3);
        apply_row_weights(block, wp);
        fill_J_block(J, m, n, roff, off_xi(i+1) + 3, block);

        /* ∂r_p/∂ba = -J_dp_ba */
        for (int k = 0; k < 9; k++) block[k] = -kf1->preint.J_dp_ba[k];
        apply_row_weights(block, wp);
        fill_J_block(J, m, n, roff, ctx->ba_off, block);

        /* ∂r_p/∂bg = -J_dp_bg */
        for (int k = 0; k < 9; k++) block[k] = -kf1->preint.J_dp_bg[k];
        apply_row_weights(block, wp);
        fill_J_block(J, m, n, roff, ctx->bg_off, block);

        ri += 9;
    }

    /* ===== PnP 视觉因子 (i ≥ 1) ===== */
    for (int i = 1; i < n_kf; i++) {
        int ik = idx(w, i);
        const window_kf_t *kf = &w->kf[ik];

        double RT_i[9], R_err[9];
        mat_transpose(R_kf[i], RT_i, 3, 3);
        mat_mul(RT_i, kf->R_meas, R_err, 3, 3, 3);

        double r_vis_R[3];
        mat33_log_so3(R_err, r_vis_R);
        double Jr_inv_vis[9];
        mat33_right_jacobian_inv(r_vis_R, Jr_inv_vis);

        double wr = 1.0 / w->sigma_vis_r;
        double wt = 1.0 / w->sigma_vis_t;
        double wR[3] = {wr, wr, wr};
        double wT[3] = {wt, wt, wt};

        double block[9], tmp[9];

        /* ∂r_vis_R/∂φ_i = -Jr^{-1}(r_vis_R)·Jr(φ_i) */
        mat_mul(Jr_inv_vis, Jr_kf[i], tmp, 3, 3, 3);
        for (int k = 0; k < 9; k++) block[k] = -tmp[k];
        apply_row_weights(block, wR);
        fill_J_block(J, m, n, ri, off_xi(i), block);

        /* ∂r_vis_t/∂t_i = I */
        double I[9] = {1,0,0, 0,1,0, 0,0,1};
        apply_row_weights(I, wT);
        fill_J_block(J, m, n, ri + 3, off_xi(i) + 3, I);

        ri += 6;
    }

    /* v_0 速度先验: ∂r/∂v_0 = I/σ */
    if (w->sigma_v0_prior > 0) {
        double wr0 = 1.0 / w->sigma_v0_prior;
        for (int k = 0; k < 3; k++) {
            J[ri * n + k] = wr0;       /* ∂r_v0[k]/∂v_0[k] = 1/σ */
            ri++;
        }
    }

    /* 应用 Huber IRLS 权重到 Jacobian 各行 */
    for (int row = 0; row < m; row++) {
        double wt = ctx->huber_wt[row];
        if (wt != 1.0) {
            for (int col = 0; col < n; col++)
                J[row * n + col] *= wt;
        }
    }

}

/* ---------- API 实现 ---------- */

void window_init(window_t *w, const double g[3],
                 double sigma_vis_r, double sigma_vis_t)
{
    w->n_kf = 0;
    w->start = 0;
    w->ba[0] = 0; w->ba[1] = 0; w->ba[2] = 0;
    w->bg[0] = 0; w->bg[1] = 0; w->bg[2] = 0;
    if (g) {
        w->g[0] = g[0]; w->g[1] = g[1]; w->g[2] = g[2];
    } else {
        w->g[0] = 0; w->g[1] = 0; w->g[2] = -9.81;
    }
    w->sigma_vis_r = sigma_vis_r > 0 ? sigma_vis_r : 0.01;
    w->sigma_vis_t = sigma_vis_t > 0 ? sigma_vis_t : 0.05;
    w->huber_delta = 0;            /* Huber 核默认关闭 */
    w->sigma_v0_prior = 0;         /* v0 先验默认关闭 */
    w->has_prior = 0;              /* 边际化先验默认空 */
    w->n_iter = 0;
    w->converged = 0;
    w->final_cost = 0;
}

int window_add_keyframe(window_t *w, double ts,
                        const double R_pnp[9], const double t_pnp[3],
                        const imu_preint_t *preint)
{
    if (w->n_kf >= WINDOW_MAX_KF) return -1;

    int ik = (w->start + w->n_kf) % WINDOW_MAX_KF;
    window_kf_t *kf = &w->kf[ik];

    kf->ts = ts;
    mat_copy(R_pnp, kf->R_meas, 3, 3);
    kf->t_meas[0] = t_pnp[0]; kf->t_meas[1] = t_pnp[1]; kf->t_meas[2] = t_pnp[2];

    /* 初始估计 = PnP 观测 */
    mat_copy(R_pnp, kf->R, 3, 3);
    kf->t[0] = t_pnp[0]; kf->t[1] = t_pnp[1]; kf->t[2] = t_pnp[2];
    kf->v[0] = 0; kf->v[1] = 0; kf->v[2] = 0;

    /* 深拷贝预积分 */
    kf->preint = *preint;

    /* 首个关键帧: 速度设为零 */
    if (w->n_kf == 0) {
        w->kf[ik].v[0] = 0; w->kf[ik].v[1] = 0; w->kf[ik].v[2] = 0;
    }

    w->n_kf++;
    return 0;
}

int window_optimize(window_t *w, int max_iter)
{
    int n_kf = w->n_kf;
    if (n_kf < 2) return -1;

    int n_state = 3 + (n_kf - 1) * 9 + 6;  /* v_0 + (N-1)*(ξ+v) + ba+bg */
    int n_resid = (n_kf - 1) * 15;          /* (N-1)*(9 IMU + 6 PnP) */
    if (w->sigma_v0_prior > 0) n_resid += 3;  /* v_0 速度先验 */

    /* 打包状态到 x[] */
    double x[WINDOW_STATE_DIM];

    /* v_0 */
    int i0 = idx(w, 0);
    x[0] = w->kf[i0].v[0]; x[1] = w->kf[i0].v[1]; x[2] = w->kf[i0].v[2];

    /* 关键帧 1..N-1 */
    for (int i = 1; i < n_kf; i++) {
        int ik = idx(w, i);
        const window_kf_t *kf = &w->kf[ik];
        se3_log(kf->R, kf->t, x + off_xi(i));
        const double *v = kf->v;
        int vo = off_v(i);
        x[vo+0] = v[0]; x[vo+1] = v[1]; x[vo+2] = v[2];
    }

    /* 偏置 */
    int ba_off = off_ba(n_kf), bg_off = off_bg(n_kf);
    x[ba_off+0] = w->ba[0]; x[ba_off+1] = w->ba[1]; x[ba_off+2] = w->ba[2];
    x[bg_off+0] = w->bg[0]; x[bg_off+1] = w->bg[1]; x[bg_off+2] = w->bg[2];

    /* 保存偏置线性化点 (预积分使用的偏置) */
    resid_ctx_t ctx;
    ctx.w = w;
    ctx.n_state = n_state;
    ctx.n_resid = n_resid;
    ctx.n_kf = n_kf;
    ctx.ba_off = ba_off;
    ctx.bg_off = bg_off;
    ctx.ba_lin[0] = w->kf[idx(w, 1)].preint.ba[0];
    ctx.ba_lin[1] = w->kf[idx(w, 1)].preint.ba[1];
    ctx.ba_lin[2] = w->kf[idx(w, 1)].preint.ba[2];
    ctx.bg_lin[0] = w->kf[idx(w, 1)].preint.bg[0];
    ctx.bg_lin[1] = w->kf[idx(w, 1)].preint.bg[1];
    ctx.bg_lin[2] = w->kf[idx(w, 1)].preint.bg[2];

    /* 调用 LM */
    lm_config_t cfg;
    cfg.max_iter = max_iter;
    cfg.tau = 1e-3;
    cfg.eps_grad = 1e-6;
    cfg.eps_step = 1e-6;
    cfg.eps_cost = 1e-6;
    cfg.lambda_init = -1.0;  /* 自动 */

    lm_result_t result;
    int ret = lm_solve(x, n_resid, n_state,
                       window_residual, window_jacobian,
                       &ctx, &cfg, &result);

    w->n_iter = result.iterations;
    w->converged = result.converged;
    w->final_cost = result.final_cost;

    /* 解包状态 */
    w->kf[i0].v[0] = x[0]; w->kf[i0].v[1] = x[1]; w->kf[i0].v[2] = x[2];

    for (int i = 1; i < n_kf; i++) {
        int ik = idx(w, i);
        se3_exp(x + off_xi(i), w->kf[ik].R, w->kf[ik].t);
        int vo = off_v(i);
        w->kf[ik].v[0] = x[vo+0];
        w->kf[ik].v[1] = x[vo+1];
        w->kf[ik].v[2] = x[vo+2];
    }

    w->ba[0] = x[ba_off+0]; w->ba[1] = x[ba_off+1]; w->ba[2] = x[ba_off+2];
    w->bg[0] = x[bg_off+0]; w->bg[1] = x[bg_off+1]; w->bg[2] = x[bg_off+2];

    return ret;
}

int window_drop_oldest(window_t *w)
{
    if (w->n_kf <= 0) return -1;
    w->start = (w->start + 1) % WINDOW_MAX_KF;
    w->n_kf--;
    return 0;
}

void window_get_pose(const window_t *w, double R[9], double t[3], double v[3])
{
    int ik = idx(w, w->n_kf - 1);
    mat_copy(w->kf[ik].R, R, 3, 3);
    t[0] = w->kf[ik].t[0]; t[1] = w->kf[ik].t[1]; t[2] = w->kf[ik].t[2];
    v[0] = w->kf[ik].v[0]; v[1] = w->kf[ik].v[1]; v[2] = w->kf[ik].v[2];
}

void window_get_biases(const window_t *w, double ba[3], double bg[3])
{
    ba[0] = w->ba[0]; ba[1] = w->ba[1]; ba[2] = w->ba[2];
    bg[0] = w->bg[0]; bg[1] = w->bg[1]; bg[2] = w->bg[2];
}

void window_set_huber_delta(window_t *w, double delta)
{
    w->huber_delta = delta > 0 ? delta : 0;
}

void window_set_v0_prior(window_t *w, double sigma)
{
    w->sigma_v0_prior = sigma > 0 ? sigma : 0;
}

int window_marginalize_oldest(window_t *w)
{
    /* Schur 补边际化尚未实现 (v2 计划)。
     * 需要对最老关键帧涉及的所有残差构建 Hessian,
     * 计算 H* = H_bb - H_ba·H_aa^{-1}·H_ab,
     * 将结果存为 prior 因子。 */
    (void)w;
    return -1;
}
