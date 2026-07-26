#ifndef TINY_WINDOW_H
#define TINY_WINDOW_H

/* ============================================================
 * tiny_window — 滑动窗口紧耦合优化器
 *
 * 纯 C99, 栈分配, 无 malloc。依赖 tiny_linalg + tiny_opt + tiny_imu。
 *
 * 维持最近 N 个关键帧的状态 (位姿+速度) 和 IMU 偏置,
 * 将 IMU 预积分约束 + PnP 视觉约束放入同一个 LM 问题求解。
 *
 * 状态参数化 (每关键帧):
 *   ξ ∈ se(3) = [rx,ry,rz, tx,ty,tz]  — Lie 代数位姿 (6 维)
 *   v ∈ R³                            — 世界系速度 (3 维)
 *
 * 共享状态: ba[3], bg[3]   — IMU 偏置
 *
 * 首个关键帧的位姿固定 (消除规范自由度), 速度参与估计。
 * ============================================================ */

#include "../tiny_imu/tiny_imu.h"

#define WINDOW_MAX_KF      5
#define WINDOW_STATE_DIM   (9 * WINDOW_MAX_KF)   /* 45 for N=5 */
#define WINDOW_MAX_RESID   (15 * (WINDOW_MAX_KF - 1) + 6)  /* 66 for N=5 */

typedef struct {
    double ts;                       /* 关键帧时间戳               */
    double R_meas[9], t_meas[3];     /* PnP 观测 (世界系)          */
    double R[9], t[3], v[3];         /* 当前估计 (优化后更新)      */
    imu_preint_t preint;             /* 从前帧到本帧的预积分       */
} window_kf_t;

typedef struct {
    window_kf_t kf[WINDOW_MAX_KF];
    int         n_kf;                /* 当前关键帧数量 (0..max)    */
    int         start;               /* 最老关键帧在 kf[] 的下标   */

    double      ba[3], bg[3];        /* IMU 偏置估计               */
    double      g[3];                /* 世界系重力 [0,0,-9.81]    */
    double      sigma_vis_r;         /* PnP 旋转噪声 (rad)         */
    double      sigma_vis_t;         /* PnP 平移噪声 (m)           */
    double      huber_delta;         /* Huber 核阈值 (>0 启用, 0 禁用) */
    double      sigma_v0_prior;      /* v_0 速度先验 std (m/s), 0=禁用  */

    /* 边际化先验 (预留, 尚未实现) */
    int         has_prior;
    double      H_prior[WINDOW_STATE_DIM * WINDOW_STATE_DIM];
    double      b_prior[WINDOW_STATE_DIM];
    double      x_prior[WINDOW_STATE_DIM];

    /* 优化结果统计 */
    int         n_iter;
    int         converged;
    double      final_cost;
} window_t;

/* ---------- API ---------- */

/* 初始化滑动窗口。
 * g:    世界系重力向量 (如 {0,0,-9.81})
 *      传 NULL 则默认 {0,0,-9.81}
 * sr,st: PnP 视觉噪声 std (rad, m), 用于残差加权       */
void window_init(window_t *w, const double g[3],
                 double sigma_vis_r, double sigma_vis_t);

/* 添加新关键帧。
 * ts:      时间戳
 * R_pnp,t_pnp: PnP 观测位姿 (世界系, 3×3 行优先)
 * preint:   从上一关键帧到当前帧的 IMU 预积分 (所有权拷贝)
 * 返回 0 成功, -1 窗口已满需先调用 window_optimize 后丢弃最老帧 */
int  window_add_keyframe(window_t *w, double ts,
                         const double R_pnp[9], const double t_pnp[3],
                         const imu_preint_t *preint);

/* 运行 LM 滑动窗口优化。
 * max_iter: 最大 LM 迭代次数 (建议 30)
 * 返回 0 收敛, -1 未收敛                  */
int  window_optimize(window_t *w, int max_iter);

/* 丢弃最老关键帧 (不边际化, 直接丢掉)。
 * 必须先调用 window_optimize 再调用此函数。
 * 返回 0 成功, -1 窗口为空                */
int  window_drop_oldest(window_t *w);

/* 设置 Huber 核阈值 (delta > 0 启用, 0 禁用)。
 * Huber 损失: ρ(r) = 0.5*r² (|r|≤δ), ρ(r) = δ·(|r|-0.5·δ) (|r|>δ)
 * 以 IRLS 方式实现: 残差/Jacobian 行乘以 √(δ/|r|) */
void window_set_huber_delta(window_t *w, double delta);

/* 设置 v_0 速度先验 std (m/s)。sigma > 0 启用, 0 禁用。
 * 对于智能车从静止启动的场景, 建议设 0.01~0.05 */
void window_set_v0_prior(window_t *w, double sigma);

/* Schur 补边际化最老关键帧 (预留接口, 尚未实现)。
 * 返回 0 成功, -1 未实现                */
int  window_marginalize_oldest(window_t *w);

/* 获取最新关键帧的优化后位姿和速度 */
void window_get_pose(const window_t *w, double R[9], double t[3], double v[3]);

/* 获取当前偏置估计 */
void window_get_biases(const window_t *w, double ba[3], double bg[3]);

#endif /* TINY_WINDOW_H */
