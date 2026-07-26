#ifndef TINY_IMU_H
#define TINY_IMU_H

/* ============================================================
 * tiny_imu — IMU 预积分 (Forster et al., TRO 2017)
 *
 * 纯 C99, 栈分配, 无 malloc。依赖 tiny_linalg。
 * 在连续两个关键帧之间对 IMU 采样做预积分:
 *   ΔR = Π Exp((ω_m - b_g)·Δt)
 *   Δv = Σ ΔR·(a_m - b_a)·Δt
 *   Δp = Σ Σ ΔR·(a_m - b_a)·Δt²
 * 同时传播 9×9 误差协方差和对偏置的 Jacobian。
 * ============================================================ */

/* 误差状态维度: δθ(SO(3)), δv(R³), δp(R³) */
#define IMU_PREINT_ERR_DIM 9
/* 测量噪声维度: η_g(3), η_a(3) */
#define IMU_PREINT_MEAS_DIM 6

typedef struct {
    /* 预积分量 */
    double dR[9];       /* 旋转增量, SO(3), 3×3 行优先 */
    double dv[3];       /* 速度增量 */
    double dp[3];       /* 位移增量 */
    double dt;          /* 累计时间间隔 */

    /* 9×9 误差状态协方差 (δθ, δv, δp), 行优先 */
    double cov[IMU_PREINT_ERR_DIM * IMU_PREINT_ERR_DIM];

    /* 偏置 Jacobians (各 3×3), 用于一阶偏置修正 */
    double J_dR_bg[9];  /* ∂ΔR/∂b_g, 3×3 行优先 */
    double J_dv_ba[9];  /* ∂Δv/∂b_a */
    double J_dv_bg[9];  /* ∂Δv/∂b_g */
    double J_dp_ba[9];  /* ∂Δp/∂b_a */
    double J_dp_bg[9];  /* ∂Δp/∂b_g */

    /* IMU 噪声参数 (连续时间) */
    double sigma_a;     /* 加速度计噪声密度 (m/s²/√Hz)   */
    double sigma_g;     /* 陀螺仪噪声密度   (rad/s/√Hz)  */
    double sigma_ba;    /* 加速度计偏置随机游走 (m/s³/√Hz) */
    double sigma_bg;    /* 陀螺仪偏置随机游走   (rad/s²/√Hz) */

    /* 当前偏置估计 (预积分期间视为常数) */
    double ba[3];
    double bg[3];
} imu_preint_t;

/* 初始化预积分量。噪声参数 + 初始偏置。
 * 内部将 ΔR 设为单位阵, Δv/Δp 清零, 协方差/Jacobian 清零。 */
void imu_preint_init(imu_preint_t *p,
                     double sigma_a, double sigma_g,
                     double sigma_ba, double sigma_bg,
                     const double ba[3], const double bg[3]);

/* 处理一次 IMU 采样。前向 Euler 积分 + 误差状态协方差传播。
 * acc: 加速度计读数 (m/s², 机体坐标系)
 * gyr: 陀螺仪读数 (rad/s, 机体坐标系)
 * dt:  距上次采样的时间间隔 (s) */
void imu_preint_update(imu_preint_t *p,
                       const double acc[3], const double gyr[3],
                       double dt);

/* 重置预积分量, 准备下一段关键帧间隔。
 * 保留噪声参数不变, 偏置更新为新值。 */
void imu_preint_reset(imu_preint_t *p,
                      const double ba[3], const double bg[3]);

/* 提取当前累积的预积分量 */
void imu_preint_get_delta(const imu_preint_t *p,
                          double dR[9], double dv[3], double dp[3]);

/* 一阶偏置修正: 用新偏置修正预积分量, 避免重新积分。
 * 基于 Jacobian 的线性化修正:
 *   ΔR(bg_new) ≈ ΔR(bg) · Exp(J_dR_bg · δbg)
 *   Δv(b_new)  ≈ Δv(b)  + J_dv_ba·δba + J_dv_bg·δbg
 *   Δp(b_new)  ≈ Δp(b)  + J_dp_ba·δba + J_dp_bg·δbg
 * 其中 δba = ba_new-ba, δbg = bg_new-bg */
void imu_preint_bias_correct(imu_preint_t *p,
                             const double ba_new[3],
                             const double bg_new[3]);

#endif /* TINY_IMU_H */
