/* ============================================================
 * test_window.c — 滑动窗口紧耦合优化器测试
 *
 * 模拟: 车辆在 XY 平面做圆周运动 (半径 5m, 角速度 1 rad/s)
 *       IMU 200Hz + 噪声, PnP 5Hz + 噪声
 *       滑动窗口 5 关键帧, LM 优化
 *       验证: 优化后位姿比原始 PnP 更接近真值
 *
 * 编译 (x86 PC):
 *   gcc -std=c99 -Wall -Wextra -O2 -o test_window
 *       tiny_window/test_window.c tiny_window/tiny_window.c
 *       tiny_imu/tiny_imu.c tiny_opt/tiny_opt.c
 *       tiny_linalg/tiny_linalg.c -lm
 * ============================================================ */

#include "stdio.h"
#include "math.h"
#include "tiny_window.h"
#include "../tiny_linalg/tiny_linalg.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL [%d]: %s\n", __LINE__, msg); } \
} while(0)

/* ================================================================
 * 真值轨迹: XY 平面圆周运动
 *   p(t) = [R*cos(ωt), R*sin(ωt), 0]
 *   v(t) = [-Rω*sin(ωt), Rω*cos(ωt), 0]
 *   a(t) = [-Rω²*cos(ωt), -Rω²*sin(ωt), 0]
 *
 * 机体系与世界系一致 (无旋转), 机体加速度 = 世界加速度 (减去重力)
 * ================================================================ */

static void true_pose(double t, double R_t[9], double p_t[3],
                       double v_t[3], double a_world[3])
{
    double R = 5.0, omega = 1.0;
    double theta = omega * t;
    double ct = cos(theta), st = sin(theta);

    /* 世界系位置 */
    p_t[0] = R * ct;
    p_t[1] = R * st;
    p_t[2] = 0.0;

    /* 世界系速度 */
    v_t[0] = -R * omega * st;
    v_t[1] =  R * omega * ct;
    v_t[2] =  0.0;

    /* 世界系加速度 (向心) */
    a_world[0] = -R * omega * omega * ct;
    a_world[1] = -R * omega * omega * st;
    a_world[2] =  0.0;

    /* 机体朝向: 始终指向切向 (车辆前进方向) */
    /* body X = 切向 = [-st, ct, 0], body Y = 径向 = [-ct, -st, 0], body Z = [0,0,1] */
    double bx[3] = {-st, ct, 0};
    double by[3] = {-ct, -st, 0};
    double bz[3] = {0, 0, 1};

    R_t[0] = bx[0]; R_t[1] = by[0]; R_t[2] = bz[0];
    R_t[3] = bx[1]; R_t[4] = by[1]; R_t[5] = bz[1];
    R_t[6] = bx[2]; R_t[7] = by[2]; R_t[8] = bz[2];
}

/* 简单随机数 */
static unsigned lcg = 42;
static double randf(void) {
    lcg = lcg * 1103515245U + 12345U;
    return (double)(lcg & 0x7FFFFFFF) / 2147483648.0;
}
static double randn(void) {
    double u1 = randf() + 1e-30, u2 = randf();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* 从旋转矩阵提取 yaw 角 (绕 Z) */
static double extract_yaw(const double R[9])
{
    return atan2(R[3], R[0]);  /* R[3]=R10, R[0]=R00 */
}

/* ================================================================ */

static void test_sliding_window(void)
{
    printf("Test: sliding window (5 kf, circular traj, 5s) ...\n");

    double g[3] = {0, 0, -9.81};
    double sigma_vis_r = 0.05;   /* ~3 deg */
    double sigma_vis_t = 0.10;   /* 10 cm */

    window_t w;
    window_init(&w, g, sigma_vis_r, sigma_vis_t);

    /* IMU 噪声 */
    double sigma_g = 0.005;   /* gyro noise rad/s/√Hz */
    double sigma_a = 0.05;    /* accel noise m/s²/√Hz */

    double dt_imu = 0.005;    /* 200 Hz */
    double dt_kf  = 0.2;      /* 5 Hz keyframe */
    int kf_steps = (int)(dt_kf / dt_imu + 0.5);  /* 40 imu per kf */
    double duration = 5.0;
    int n_kf_total = (int)(duration / dt_kf);     /* 25 keyframes */

    int n_windows = 0;
    double sum_pos_err_before = 0, sum_pos_err_after = 0;
    double sum_yaw_err_before = 0, sum_yaw_err_after = 0;

    for (int kf_idx = 0; kf_idx < n_kf_total; kf_idx++) {
        double t_kf = (kf_idx + 1) * dt_kf;

        /* --- 生成 IMU 预积分 (从前一关键帧到当前帧) --- */
        imu_preint_t preint;
        double ba0[3] = {0,0,0}, bg0[3] = {0,0,0};
        imu_preint_init(&preint, sigma_a, sigma_g, 0, 0, ba0, bg0);

        for (int s = 0; s < kf_steps; s++) {
            double t = t_kf - dt_kf + (s + 1) * dt_imu;

            double R_t[9], p_t[3], v_t[3], a_w[3];
            true_pose(t, R_t, p_t, v_t, a_w);

            double RT[9];
            mat_transpose(R_t, RT, 3, 3);
            double a_rel[3] = {a_w[0]-g[0], a_w[1]-g[1], a_w[2]-g[2]};
            double acc_body[3];
            mat_vec_mul(RT, a_rel, acc_body, 3, 3);

            double gyr[3] = {0, 0, 1.0};

            double sa_d = sigma_a / sqrt(dt_imu);
            double sg_d = sigma_g / sqrt(dt_imu);
            double acc_n[3] = {acc_body[0]+randn()*sa_d,
                               acc_body[1]+randn()*sa_d,
                               acc_body[2]+randn()*sa_d};
            double gyr_n[3] = {gyr[0]+randn()*sg_d,
                               gyr[1]+randn()*sg_d,
                               gyr[2]+randn()*sg_d};

            imu_preint_update(&preint, acc_n, gyr_n, dt_imu);
        }

        /* --- 生成带噪声的 PnP 观测 --- */
        double R_true[9], p_true[3], v_true[3], a_true[3];
        true_pose(t_kf, R_true, p_true, v_true, a_true);

        double R_noise[9];
        double noise_rot[3] = {randn()*sigma_vis_r,
                               randn()*sigma_vis_r,
                               randn()*sigma_vis_r};
        mat33_exp_so3(noise_rot, R_noise);
        double R_pnp[9];
        mat_mul(R_true, R_noise, R_pnp, 3, 3, 3);

        double t_pnp[3] = {p_true[0] + randn()*sigma_vis_t,
                           p_true[1] + randn()*sigma_vis_t,
                           p_true[2] + randn()*sigma_vis_t};

        /* --- 添加到窗口, 若满则先优化+丢弃最老帧 --- */
        if (window_add_keyframe(&w, t_kf, R_pnp, t_pnp, &preint) != 0) {
            window_optimize(&w, 30);
            window_drop_oldest(&w);
            window_add_keyframe(&w, t_kf, R_pnp, t_pnp, &preint);
        }

        /* 每轮优化后统计误差: 比较最新关键帧优化前后 vs 真值 */
        if (w.n_kf >= 3) {
            window_optimize(&w, 30);
            n_windows++;

            double R_before[9], t_before[3];
            mat_copy(R_pnp, R_before, 3, 3);
            t_before[0] = t_pnp[0];
            t_before[1] = t_pnp[1];
            t_before[2] = t_pnp[2];

            double R_after[9], t_after[3], v_after[3];
            window_get_pose(&w, R_after, t_after, v_after);

            double R_t[9], p_t[3], v_t[3], a_t[3];
            true_pose(t_kf, R_t, p_t, v_t, a_t);

            double err_before_pos = sqrt(
                (t_before[0]-p_t[0])*(t_before[0]-p_t[0]) +
                (t_before[1]-p_t[1])*(t_before[1]-p_t[1]) +
                (t_before[2]-p_t[2])*(t_before[2]-p_t[2]));
            double err_after_pos = sqrt(
                (t_after[0]-p_t[0])*(t_after[0]-p_t[0]) +
                (t_after[1]-p_t[1])*(t_after[1]-p_t[1]) +
                (t_after[2]-p_t[2])*(t_after[2]-p_t[2]));

            double yaw_true = extract_yaw(R_t);
            double yaw_before = extract_yaw(R_before);
            double yaw_after = extract_yaw(R_after);

            double err_before_yaw = fabs(yaw_before - yaw_true);
            double err_after_yaw = fabs(yaw_after - yaw_true);

            sum_pos_err_before += err_before_pos;
            sum_pos_err_after  += err_after_pos;
            sum_yaw_err_before += err_before_yaw;
            sum_yaw_err_after  += err_after_yaw;
        }
    }

    double avg_pos_before = sum_pos_err_before / n_windows;
    double avg_pos_after  = sum_pos_err_after  / n_windows;
    double avg_yaw_before = sum_yaw_err_before / n_windows;
    double avg_yaw_after  = sum_yaw_err_after  / n_windows;

    printf("  windows optimized: %d\n", n_windows);
    printf("  avg pos error: before=%.4fm  after=%.4fm  (%.0f%% improvement)\n",
           avg_pos_before, avg_pos_after,
           (1.0 - avg_pos_after/avg_pos_before) * 100.0);
    printf("  avg yaw error: before=%.4frad after=%.4frad (%.0f%% improvement)\n",
           avg_yaw_before, avg_yaw_after,
           (1.0 - avg_yaw_after/avg_yaw_before) * 100.0);

    CHECK(avg_pos_after < avg_pos_before, "pos: optimized better than raw PnP");
    CHECK(avg_yaw_after < avg_yaw_before, "yaw: optimized better than raw PnP");
}

/* ================================================================
 * 测试 2: 最小窗口 (3 关键帧) 正确性
 *
 * N=2 时 n_state=18 > n_resid=15 (欠定), LM 直接返回 -2。
 * 至少需要 N=3 才有充分约束: n_state=27 ≤ n_resid=30。
 * ================================================================ */
static void test_minimal_window(void)
{
    printf("Test: minimal window (3 kf, straight line) ...\n");

    double g[3] = {0, 0, -9.81};
    window_t w;
    window_init(&w, g, 0.01, 0.05);
    window_set_v0_prior(&w, 0.01);   /* 静止启动, v0≈0 强先验 */

    double ba0[3] = {0,0,0}, bg0[3] = {0,0,0};
    double dt = 0.005;
    double R_eye[9] = {1,0,0, 0,1,0, 0,0,1};
    double t0[3] = {0,0,0};

    /* --- kf_0: t=0, 静止, 位姿固定, v_0≈0 先验 --- */
    imu_preint_t preint0;
    imu_preint_init(&preint0, 0.01, 0.001, 0, 0, ba0, bg0);
    window_add_keyframe(&w, 0.0, R_eye, t0, &preint0);

    /* --- kf_1: t=0.5, 0.5s加速, 真值 p=0.25 v=1.0 --- */
    imu_preint_t preint1;
    imu_preint_init(&preint1, 0.01, 0.001, 0, 0, ba0, bg0);
    for (int k = 0; k < 100; k++) {
        double acc[3] = {2.0, 0, 9.81};
        double gyr[3] = {0, 0, 0};
        imu_preint_update(&preint1, acc, gyr, dt);
    }
    double t1_noisy[3] = {0.30, 0.02, -0.03};  /* 真值[0.25,0,0], 噪声5cm */
    window_add_keyframe(&w, 0.5, R_eye, t1_noisy, &preint1);

    /* --- kf_2: t=1.0, 再0.5s加速, 真值 p=1.0 v=2.0 --- */
    imu_preint_t preint2;
    imu_preint_init(&preint2, 0.01, 0.001, 0, 0, ba0, bg0);
    for (int k = 0; k < 100; k++) {
        double acc[3] = {2.0, 0, 9.81};
        double gyr[3] = {0, 0, 0};
        imu_preint_update(&preint2, acc, gyr, dt);
    }
    double t2_noisy[3] = {0.97, -0.02, 0.03};  /* 真值[1.0,0,0], 噪声~3cm */
    window_add_keyframe(&w, 1.0, R_eye, t2_noisy, &preint2);

    int ret = window_optimize(&w, 30);
    printf("  LM ret=%d  iter=%d  converged=%d  cost=%.6f\n",
           ret, w.n_iter, w.converged, w.final_cost);

    double R_out[9], t_out[3], v_out[3];
    window_get_pose(&w, R_out, t_out, v_out);

    printf("  kf_2: t=[%.4f,%.4f,%.4f] (P=[0.97,-0.02,0.03] T=[1.0,0,0])\n",
           t_out[0], t_out[1], t_out[2]);
    printf("  kf_2: v=[%.4f,%.4f,%.4f] (T=[2.0,0,0])\n",
           v_out[0], v_out[1], v_out[2]);

    /* 取 kf_0 的优化后速度: v0 先验应使其接近 0 */
    int ik0 = (w.start) % WINDOW_MAX_KF;
    double v0_x = w.kf[ik0].v[0];
    printf("  kf_0: v=[%.4f,%.4f,%.4f] (true=0, prior σ=0.01)\n",
           w.kf[ik0].v[0], w.kf[ik0].v[1], w.kf[ik0].v[2]);

    CHECK(ret == 0, "min: LM converged");
    CHECK(w.converged != 0, "min: converged flag set");
    CHECK(w.final_cost < 2.0, "min: cost reasonable (v0 prior + noisy PnP)");
    CHECK(fabs(v_out[0]) > 0.5, "min: kf_2 velocity non-zero (IMU used)");
    CHECK(fabs(v0_x) < 0.20, "min: v0 near zero (prior active)");
}

/* ================================================================ */

int main(void)
{
    printf("=== Sliding Window Optimization Tests ===\n\n");

    test_minimal_window();
    test_sliding_window();

    printf("\n=== %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
