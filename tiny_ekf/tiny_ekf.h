#ifndef TINY_EKF_H
#define TINY_EKF_H

/* ============================================================
 * tiny_ekf — 通用扩展卡尔曼滤波
 *
 * 纯 C99, 零依赖 (仅需 tiny_linalg), 栈分配, 无 malloc。
 * 状态/观测模型通过回调注入, 不绑死特定应用。
 * Joseph 形式协方差更新保证对称正定。
 * ============================================================ */

#define EKF_MAX_STATE  12
#define EKF_MAX_OBS    12

/* ---------- 回调类型 ---------- */

/* 状态转移: x[n] + u[?] + dt → x_pred[n], F[n*n] (行优先)
 * F 是 ∂f/∂x 在 x 处的 Jacobian
 * u 为控制输入, 含义由用户定义; 可为 NULL (纯预测, 无控制) */
typedef void (*ekf_transition_t)(const double *x, const double *u, double dt,
                                  double *x_pred, double *F, int n);

/* 观测模型: x[n] → z_pred[m], H[m*n] (行优先)
 * H 是 ∂h/∂x 在 x 处的 Jacobian */
typedef void (*ekf_observation_t)(const double *x,
                                   double *z_pred, double *H, int m, int n);

/* ---------- EKF 状态 ---------- */

typedef struct {
    int    n;      /* 状态维度 */
    double x[EKF_MAX_STATE];
    double P[EKF_MAX_STATE * EKF_MAX_STATE];
} ekf_t;

/* ---------- API ---------- */

/* 初始化 EKF
 * n:  状态维度 (≤ EKF_MAX_STATE)
 * x0: 初始状态 (n 维), 不可为 NULL
 * P0: 初始协方差 (n×n 行优先), 为 NULL 时设为单位阵 × 1e6 */
void ekf_init(ekf_t *ekf, int n, const double *x0, const double *P0);

/* 预测步
 * f:  状态转移回调 (不可为 NULL)
 * u:  控制输入 (可为 NULL)
 * dt: 时间步长
 * Q:  过程噪声协方差 (n×n 行优先, 不可为 NULL) */
void ekf_predict(ekf_t *ekf, ekf_transition_t f, const double *u, double dt,
                 const double *Q);

/* 更新步
 * h:  观测回调 (不可为 NULL)
 * z:  实际观测 (m 维)
 * R:  观测噪声协方差 (m×m 行优先)
 * m:  观测维度 (≤ EKF_MAX_OBS, m=0 时无操作直接返回 0)
 * 返回: m (观测维度)  */
int  ekf_update(ekf_t *ekf, ekf_observation_t h, const double *z,
                const double *R, int m);

#endif /* TINY_EKF_H */
