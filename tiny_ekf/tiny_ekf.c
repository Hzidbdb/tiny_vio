/*
 * @Author: garygoo
 * @Date: 2026-07-25
 * @Description: 通用扩展卡尔曼滤波 (Joseph 形式协方差更新)
 *   依赖: tiny_linalg
 */

#include "string.h"
#include "tiny_ekf.h"
#include "../tiny_linalg/tiny_linalg.h"

/* ---------- 初始化 ---------- */

void ekf_init(ekf_t *ekf, int n, const double *x0, const double *P0)
{
    ekf->n = n;
    memcpy(ekf->x, x0, (size_t)n * sizeof(double));

    if (P0) {
        memcpy(ekf->P, P0, (size_t)(n * n) * sizeof(double));
    } else {
        /* 默认: 大对角协方差, 表示初始不确定性很大 */
        mat_identity(ekf->P, n);
        for (int i = 0; i < n; i++)
            ekf->P[i * n + i] = 1e6;
    }
}

/* ---------- 预测步 ---------- */

void ekf_predict(ekf_t *ekf, ekf_transition_t f, const double *u, double dt,
                 const double *Q)
{
    int n = ekf->n;
    int nn = n * n;

    /* 工作区 */
    double F[EKF_MAX_STATE * EKF_MAX_STATE];
    double Ft[EKF_MAX_STATE * EKF_MAX_STATE];
    double temp[EKF_MAX_STATE * EKF_MAX_STATE];
    double P_temp[EKF_MAX_STATE * EKF_MAX_STATE];

    /* 1. 状态转移 + Jacobian */
    f(ekf->x, u, dt, ekf->x, F, n);

    /* 2. P = F * P * F^T + Q */
    mat_mul(F, ekf->P, temp, n, n, n);      /* temp = F * P       */
    mat_transpose(F, Ft, n, n);              /* Ft   = F^T         */
    mat_mul(temp, Ft, P_temp, n, n, n);      /* P_temp = temp * Ft */

    /* P = P_temp + Q (不通过 mat_add 避免 restrict 别名问题) */
    for (int i = 0; i < nn; i++)
        ekf->P[i] = P_temp[i] + Q[i];

    /* x 已在 f() 回调中写到 ekf->x, 完成了原地更新 */
}

/* ---------- 更新步 ---------- */

int ekf_update(ekf_t *ekf, ekf_observation_t h, const double *z,
               const double *R, int m)
{
    int n = ekf->n;

    if (m <= 0 || m > EKF_MAX_OBS) return 0;

    /* ---- 工作区 ---- */
    double H[EKF_MAX_OBS * EKF_MAX_STATE];
    double Ht[EKF_MAX_STATE * EKF_MAX_OBS];
    double HP[EKF_MAX_OBS * EKF_MAX_STATE];
    double S[EKF_MAX_OBS * EKF_MAX_OBS];
    double L[EKF_MAX_OBS * EKF_MAX_OBS];
    double inv_S[EKF_MAX_OBS * EKF_MAX_OBS];
    double PHT[EKF_MAX_STATE * EKF_MAX_OBS];
    double K[EKF_MAX_STATE * EKF_MAX_OBS];
    double KH[EKF_MAX_STATE * EKF_MAX_STATE];
    double I_KH[EKF_MAX_STATE * EKF_MAX_STATE];
    double I_KHt[EKF_MAX_STATE * EKF_MAX_STATE];
    double P_temp[EKF_MAX_STATE * EKF_MAX_STATE];
    double KR[EKF_MAX_STATE * EKF_MAX_OBS];
    double Kt[EKF_MAX_OBS * EKF_MAX_STATE];
    double KRKt[EKF_MAX_STATE * EKF_MAX_STATE];

    double z_pred[EKF_MAX_OBS];
    double y[EKF_MAX_OBS];
    double dx[EKF_MAX_STATE];
    double e[EKF_MAX_OBS];
    double inv_col[EKF_MAX_OBS];

    /* ---- 1. 观测预测 ---- */
    h(ekf->x, z_pred, H, m, n);

    /* y = z - z_pred (新息) */
    vec_remove(z, z_pred, y, m);

    /* ---- 2. 新息协方差 S = H * P * H^T + R ---- */
    mat_mul(H, ekf->P, HP, m, n, n);          /* HP = H * P       */
    mat_transpose(H, Ht, m, n);                /* Ht = H^T         */
    mat_mul(HP, Ht, S, m, n, m);               /* S  = HP * Ht     */

    /* 加 R: S += R (避免 restrict 别名) */
    {
        int mm = m * m;
        for (int i = 0; i < mm; i++)
            S[i] += R[i];
    }

    /* ---- 3. Cholesky S = L * L^T ---- */
    {
        int ret;
        for (int perturb = 0; perturb < 4; perturb++) {
            ret = mat_cholesky(S, L, m);
            if (ret == 0) break;
            /* 非正定: 对角加微小扰动恢复正定性 */
            double eps = (perturb == 0) ? 1e-10
                       : (perturb == 1) ? 1e-8
                       : (perturb == 2) ? 1e-6
                       : 1e-4;
            for (int i = 0; i < m; i++)
                S[i * m + i] += eps;
        }
        /* 若 4 次均失败, 跳过此观测 (返回无更新) */
        if (ret != 0) return m;
    }

    /* ---- 4. 求 S^{-1} (逐列回代) ---- */
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < m; i++) e[i] = 0.0;
        e[j] = 1.0;
        mat_cholesky_solve(L, e, inv_col, m);
        for (int i = 0; i < m; i++)
            inv_S[i * m + j] = inv_col[i];
    }

    /* ---- 5. Kalman 增益 K = P * H^T * inv(S) ---- */
    mat_mul(ekf->P, Ht, PHT, n, n, m);         /* PHT = P * Ht   */
    mat_mul(PHT, inv_S, K, n, m, m);            /* K   = PHT * inv_S */

    /* ---- 6. 状态更新 x = x + K * y ---- */
    mat_vec_mul(K, y, dx, n, m);
    /* vec_add 是 restrict, 不能同指针 → 循环原地加 */
    for (int i = 0; i < n; i++)
        ekf->x[i] += dx[i];

    /* ---- 7. Joseph 协方差更新 ---- */
    /* KH = K * H */
    mat_mul(K, H, KH, n, m, n);

    /* I_KH = I - KH */
    mat_identity(I_KH, n);
    /* mat_sub 是 restrict, 不能同指针 → 循环原地减 */
    {
        int nn = n * n;
        for (int i = 0; i < nn; i++)
            I_KH[i] -= KH[i];
    }

    /* I_KHt = (I - KH)^T */
    mat_transpose(I_KH, I_KHt, n, n);

    /* P_temp = (I - KH) * P */
    mat_mul(I_KH, ekf->P, P_temp, n, n, n);

    /* P = P_temp * (I - KH)^T */
    mat_mul(P_temp, I_KHt, ekf->P, n, n, n);

    /* KR = K * R */
    mat_mul(K, R, KR, n, m, m);

    /* Kt = K^T */
    mat_transpose(K, Kt, n, m);

    /* KRKt = KR * K^T */
    mat_mul(KR, Kt, KRKt, n, m, n);

    /* P += KRKt (避免 restrict 别名) */
    {
        int nn = n * n;
        for (int i = 0; i < nn; i++)
            ekf->P[i] += KRKt[i];
    }

    return m;
}
