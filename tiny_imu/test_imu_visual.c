/*
 * test_imu_visual.c — IMU 预积分可视化测试
 *
 * 生成 5 种运动轨迹的合成 IMU 数据, 运行预积分,
 * 输出 CSV 文件供 Python/matplotlib 绘图。
 *
 * 编译 (x86 PC):
 *   gcc -std=c99 -Wall -Wextra -O2 -o test_imu_visual
 *       tiny_imu/test_imu_visual.c tiny_imu/tiny_imu.c
 *       tiny_linalg/tiny_linalg.c -lm
 *
 * 运行: ./test_imu_visual
 *   输出 5 个 CSV 文件到当前目录: imu_vis_*.csv
 *   然后运行: python tools/plot_imu.py
 */

#include "stdio.h"
#include "math.h"
#include "tiny_imu.h"
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
 * 轨迹 1: 纯旋转 — 绕 Z 轴恒定角速度 2 rad/s, 运行 3 秒
 *
 * CSV: time, dR00, dR01, dR10, dR11, theta_true
 * 预期: dR = [[cos(2t), -sin(2t), 0], [sin(2t), cos(2t), 0], [0,0,1]]
 * ================================================================ */
static void test1_pure_rotation(void)
{
    printf("Test 1: pure rotation (Z axis, 2 rad/s, 3s) ...\n");

    FILE *f = fopen("imu_vis_1_rotation.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,dR00,dR01,dR10,dR11,theta_true\n");

    imu_preint_t p;
    double ba[3] = {0,0,0}, bg[3] = {0,0,0};
    imu_preint_init(&p, 0.001, 0.001, 1e-6, 1e-6, ba, bg);

    double dt = 0.005;  /* 200 Hz */
    int steps = (int)(3.0 / dt + 0.5);  /* 3 seconds, 四舍五入防浮点截断 */
    double omega_z = 2.0;

    for (int k = 0; k < steps; k++) {
        double t = k * dt;
        double gyr[3] = {0, 0, omega_z};
        double acc[3] = {0, 0, 0};

        imu_preint_update(&p, acc, gyr, dt);

        /* 每 0.05s 输出一行 (10 个采样) */
        if (k % 10 == 0) {
            double dR[9], dv[3], dp[3];
            imu_preint_get_delta(&p, dR, dv, dp);
            double theta_true = omega_z * (t + dt);
            fprintf(f, "%.4f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                    t + dt, dR[0], dR[1], dR[3], dR[4], theta_true);
        }
    }
    fclose(f);

    /* 最终验证 */
    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);
    double theta = 3.0 * omega_z;  /* 6 rad */
    CHECK(fabs(dR[0] - cos(theta)) < 1e-12, "rot: R00=cos(6)");
    CHECK(fabs(dR[1] + sin(theta)) < 1e-12, "rot: R01=-sin(6)");
    CHECK(fabs(dR[3] - sin(theta)) < 1e-12, "rot: R10=sin(6)");
    CHECK(fabs(p.dt - 3.0) < 1e-12, "rot: dt=3.0");
    printf("  wrote %d rows\n", steps / 10);
}

/* ================================================================
 * 轨迹 2: 纯平移 — 机体 X 轴恒定加速度 3 m/s², 运行 2 秒
 *
 * CSV: time, dv_x, dv_y, dp_x, dp_y, dv_x_true, dp_x_true
 * 预期: Δv_x = 3t, Δp_x = 0.5*3*t²
 * ================================================================ */
static void test2_constant_accel(void)
{
    printf("Test 2: constant acceleration (X axis, 3 m/s^2, 2s) ...\n");

    FILE *f = fopen("imu_vis_2_accel.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,dv_x,dv_y,dp_x,dp_y,dv_x_true,dp_x_true\n");

    imu_preint_t p;
    double ba[3] = {0,0,0}, bg[3] = {0,0,0};
    imu_preint_init(&p, 0.001, 0.001, 1e-6, 1e-6, ba, bg);

    double dt = 0.005;
    int steps = (int)(2.0 / dt + 0.5);  /* 防浮点截断 */
    double a_x = 3.0;

    for (int k = 0; k < steps; k++) {
        double t = k * dt;
        double acc[3] = {a_x, 0, 0};
        double gyr[3] = {0, 0, 0};

        imu_preint_update(&p, acc, gyr, dt);

        if (k % 10 == 0) {
            double dR[9], dv[3], dp[3];
            imu_preint_get_delta(&p, dR, dv, dp);
            double T = t + dt;
            double dv_true = a_x * T;
            double dp_true = 0.5 * a_x * T * T;
            fprintf(f, "%.4f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                    T, dv[0], dv[1], dp[0], dp[1], dv_true, dp_true);
        }
    }
    fclose(f);

    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);
    double T = 2.0;
    CHECK(fabs(dv[0] - a_x * T) < 1e-12, "accel: dv_x=6.0");
    CHECK(fabs(dp[0] - 0.5 * a_x * T * T) < 1e-12, "accel: dp_x=6.0");
    printf("  wrote %d rows\n", steps / 10);
}

/* ================================================================
 * 轨迹 3: 圆周运动 — 机体绕 Z 旋转 + X 轴向心加速度
 *
 * 半径 R=2m, 角速度 ω=1.5 rad/s, 运行 4 秒
 * 机体 X 始终指向切线方向, Z 指向旋转轴
 * 在机体系中: ω_body = [0, 0, 1.5], a_body = [0, ω²R, 0] (向心在Y)
 *
 * 对于匀速圆周运动, 切向加速度为0, 向心加速度 = ω²R
 * 但这里我们模拟的是: 机体固定在旋转体上, X=切向, Y=径向(向心)
 * 不对, 让我们简化: X=前进方向, 加速度在X方向 = 0 (匀速)
 * 向心加速度在Y方向 = ω²R
 *
 * CSV: time, dR00..dR08, dv[0..2], dp[0..2], pos_true_x,y
 * ================================================================ */
static void test3_circular_motion(void)
{
    printf("Test 3: circular motion (R=2m, ω=1.5 rad/s, 4s) ...\n");

    FILE *f = fopen("imu_vis_3_circular.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,dR00,dR11,dR01,dv_x,dv_y,dp_x,dp_y,pos_true_x,pos_true_y\n");

    imu_preint_t p;
    double ba[3] = {0,0,0}, bg[3] = {0,0,0};
    imu_preint_init(&p, 0.001, 0.001, 1e-6, 1e-6, ba, bg);

    double dt = 0.005;
    int steps = (int)(4.0 / dt + 0.5);  /* 防浮点截断 */
    double omega = 1.5;
    double R = 2.0;

    for (int k = 0; k < steps; k++) {
        double t = k * dt;

        /* 在机体系中: 绕Z轴旋转, 向心加速度在Y方向 */
        double gyr[3] = {0, 0, omega};
        double a_c = omega * omega * R;  /* 向心加速度大小 */
        double acc[3] = {0, a_c, 0};

        imu_preint_update(&p, acc, gyr, dt);

        if (k % 20 == 0) {  /* 每 0.1s */
            double dR[9], dv[3], dp[3];
            imu_preint_get_delta(&p, dR, dv, dp);
            double T = t + dt;
            double theta = omega * T;
            double px = R * sin(theta);
            double py = -R * cos(theta) + R;  /* 起点在 (0,0), 圆心在 (0,R) */

            fprintf(f, "%.4f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                    T, dR[0], dR[4], dR[1],
                    dv[0], dv[1], dp[0], dp[1], px, py);
        }
    }
    fclose(f);

    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);
    double theta = 4.0 * omega;  /* 6 rad */
    CHECK(fabs(dR[0] - cos(theta)) < 1e-12, "circ: R00=cos(6)");
    CHECK(fabs(p.dt - 4.0) < 1e-12, "circ: dt=4.0");
    printf("  wrote %d rows\n", steps / 20);
}

/* ================================================================
 * 轨迹 4: 协方差增长 — 恒定噪声, 运行 5 秒
 *
 * CSV: time, cov_rot_trace, cov_vel_trace, cov_pos_trace
 * 展示 δθ/δv/δp 三个子块的方差如何随时间累积
 * ================================================================ */
static void test4_covariance(void)
{
    printf("Test 4: covariance growth (5s, 200Hz) ...\n");

    FILE *f = fopen("imu_vis_4_covariance.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,var_dR00,var_dR11,var_dv_x,var_dv_y,var_dp_x,var_dp_y,"
            "cov_trace_rot,cov_trace_vel,cov_trace_pos\n");

    imu_preint_t p;
    double ba[3] = {0,0,0}, bg[3] = {0,0,0};
    /* 用较大噪声使协方差增长明显 */
    imu_preint_init(&p, 0.05, 0.02, 1e-4, 1e-4, ba, bg);

    double dt = 0.005;
    int steps = (int)(5.0 / dt + 0.5);  /* 防浮点截断 */
    /* 轻微运动 */
    double acc[3] = {0.5, -0.3, 0.2};
    double gyr[3] = {0.1, -0.05, 0.3};

    for (int k = 0; k < steps; k++) {
        imu_preint_update(&p, acc, gyr, dt);

        if (k % 20 == 0) {
            double T = (k + 1) * dt;

            /* 9x9 cov 子块 trace:
             * cov[0:3, 0:3] = δθ covariance → diag indices 0,10,20
             * cov[3:6, 3:6] = δv covariance → diag indices 30,40,50
             * cov[6:9, 6:9] = δp covariance → diag indices 60,70,80
             * cov index = i*9 + j */
            double trace_rot = p.cov[0*9+0] + p.cov[1*9+1] + p.cov[2*9+2];
            double trace_vel = p.cov[3*9+3] + p.cov[4*9+4] + p.cov[5*9+5];
            double trace_pos = p.cov[6*9+6] + p.cov[7*9+7] + p.cov[8*9+8];

            fprintf(f, "%.4f,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    T,
                    p.cov[0*9+0], p.cov[1*9+1],  /* var(δθ_x), var(δθ_y) */
                    p.cov[3*9+3], p.cov[4*9+4],  /* var(δv_x), var(δv_y) */
                    p.cov[6*9+6], p.cov[7*9+7],  /* var(δp_x), var(δp_y) */
                    trace_rot, trace_vel, trace_pos);
        }
    }
    fclose(f);
    printf("  wrote %d rows\n", steps / 20);

    /* 协方差应该持续增长 */
    double trace = 0;
    for (int i = 0; i < 9; i++) trace += p.cov[i*9 + i];
    CHECK(trace > 1e-6, "cov: trace > 0 after 5s");
}

/* ================================================================
 * 轨迹 5: 偏置修正 — 先用错误偏置积分, 再修正对比
 *
 * 模拟: 初始化时偏置估计有误差, EKF 运行一段后得到更好的偏置
 * 估计, 展示一阶修正前 vs 修正后 vs 参考值 (正确偏置积分) 的差距。
 *
 * CSV: time, dv_x_wrong, dv_x_ref, dp_x_wrong, dp_x_ref
 * 另输出修正瞬间的前后快照
 * ================================================================ */
static void test5_bias_correction(void)
{
    printf("Test 5: bias correction (3s wrong bias, then correct) ...\n");

    /* 真实运动: 绕 Y 轴慢转 + X 方向加速 */
    double dt = 0.005;
    int steps = (int)(3.0 / dt + 0.5);  /* 防浮点截断 */

    /* 参考积分器 (正确偏置 = 零) */
    imu_preint_t ref;
    double ba0[3] = {0,0,0}, bg0[3] = {0,0,0};
    imu_preint_init(&ref, 0.001, 0.001, 1e-6, 1e-6, ba0, bg0);

    /* 错误偏置积分器 */
    imu_preint_t wrong;
    double ba_bad[3] = {0.08, -0.03, 0.01};
    double bg_bad[3] = {0.002, 0.001, -0.003};
    imu_preint_init(&wrong, 0.001, 0.001, 1e-6, 1e-6, ba_bad, bg_bad);

    FILE *f = fopen("imu_vis_5_bias.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,dv_x_wrong,dv_x_ref,dp_x_wrong,dp_x_ref,"
            "dv_x_fixed,dp_x_fixed\n");

    int fixed = 0;
    imu_preint_t corrected;

    for (int k = 0; k < steps; k++) {
        double t = k * dt;
        double gyr[3] = {0, 0.5*sin(0.7*t), 0.3};
        double acc[3] = {2.0*cos(0.5*t) + 1.0, 0.5, 0.1*sin(t)};

        imu_preint_update(&ref, acc, gyr, dt);
        imu_preint_update(&wrong, acc, gyr, dt);

        /* 在 1.5 秒时做修正: 复制 wrong → corrected, 然后偏置修正 */
        if (!fixed && t >= 1.5) {
            corrected = wrong;
            imu_preint_bias_correct(&corrected, ba0, bg0);
            fixed = 1;

            /* 输出修正时刻快照 */
            double dR_w[9], dv_w[3], dp_w[3];
            double dR_r[9], dv_r[3], dp_r[3];
            double dR_c[9], dv_c[3], dp_c[3];
            imu_preint_get_delta(&wrong, dR_w, dv_w, dp_w);
            imu_preint_get_delta(&ref, dR_r, dv_r, dp_r);
            imu_preint_get_delta(&corrected, dR_c, dv_c, dp_c);

            FILE *fs = fopen("imu_vis_5_bias_snapshot.csv", "w");
            if (fs) {
                fprintf(fs, "label,dv_x,dv_y,dv_z,dp_x,dp_y,dp_z\n");
                fprintf(fs, "wrong,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        dv_w[0], dv_w[1], dv_w[2], dp_w[0], dp_w[1], dp_w[2]);
                fprintf(fs, "ref,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        dv_r[0], dv_r[1], dv_r[2], dp_r[0], dp_r[1], dp_r[2]);
                fprintf(fs, "corrected,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        dv_c[0], dv_c[1], dv_c[2], dp_c[0], dp_c[1], dp_c[2]);
                fclose(fs);
            }
        }

        /* 修正后, corrected 也继续接收 IMU 数据 */
        if (fixed)
            imu_preint_update(&corrected, acc, gyr, dt);

        if (k % 10 == 0) {
            double dR_w[9], dv_w[3], dp_w[3];
            double dR_r[9], dv_r[3], dp_r[3];
            imu_preint_get_delta(&wrong, dR_w, dv_w, dp_w);
            imu_preint_get_delta(&ref, dR_r, dv_r, dp_r);

            double dv_x_fix = 0, dp_x_fix = 0;
            if (fixed) {
                double dR_c[9], dv_c[3], dp_c[3];
                imu_preint_get_delta(&corrected, dR_c, dv_c, dp_c);
                dv_x_fix = dv_c[0];
                dp_x_fix = dp_c[0];
            }

            fprintf(f, "%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    t + dt,
                    dv_w[0], dv_r[0], dp_w[0], dp_r[0],
                    dv_x_fix, dp_x_fix);
        }
    }
    fclose(f);

    /* 验证最终时刻 corrected 接近 ref (一阶修正 + 后续继续积分) */
    double dR_c[9], dv_c[3], dp_c[3];
    double dR_r[9], dv_r[3], dp_r[3];
    imu_preint_get_delta(&corrected, dR_c, dv_c, dp_c);
    imu_preint_get_delta(&ref, dR_r, dv_r, dp_r);
    double dv_err = sqrt((dv_c[0]-dv_r[0])*(dv_c[0]-dv_r[0])
                       + (dv_c[1]-dv_r[1])*(dv_c[1]-dv_r[1])
                       + (dv_c[2]-dv_r[2])*(dv_c[2]-dv_r[2]));
    /* 一阶修正 + 后续 Euler 积分, 容许 ~0.1 m/s 偏差 */
    CHECK(dv_err < 0.2, "bias: dv error < 0.2 after correction");
    printf("  dv error after correction + continued integration: %.6f\n", dv_err);
    printf("  wrote %d rows\n", steps / 10);
}

/* ================================================================
 * 轨迹 6: 真实噪声 — 给 IMU 加高斯噪声, 跑 5 条 Monte Carlo
 *
 * 使用与测试 1 相同的纯旋转轨迹 (Z 轴 2 rad/s, 3s),
 * 加速度计: σ_a = 0.01 m/s²/√Hz  → 离散 σ = 0.01 / √dt
 * 陀螺仪:   σ_g = 0.001 rad/s/√Hz → 离散 σ = 0.001 / √dt
 *
 * CSV: time, dR00_true, dR00_mean, dR00_2sigma_upper, dR00_2sigma_lower, ...
 * ================================================================ */

/* 简单 LCG 伪随机数, 返回 [0, 1) */
static unsigned lcg_state = 1;
static double lcg_rand(void)
{
    lcg_state = lcg_state * 1103515245U + 12345U;
    return (double)(lcg_state & 0x7FFFFFFF) / 2147483648.0;
}

/* Box-Muller: N(0,1) */
static double randn(void)
{
    double u1 = lcg_rand() + 1e-30;
    double u2 = lcg_rand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void test6_noisy_rotation(void)
{
    printf("Test 6: noisy rotation (5 Monte Carlo trials, 3s) ...\n");

    double dt = 0.005;
    int steps = (int)(3.0 / dt + 0.5);
    double omega_z = 2.0;

    /* 连续时间噪声密度 → 离散采样噪声标准差 */
    double sigma_g = 0.001;   /* rad/s/√Hz */
    double sigma_a = 0.01;    /* m/s²/√Hz */
    double sg_disc = sigma_g / sqrt(dt);  /* 离散噪声 std */
    double sa_disc = sigma_a / sqrt(dt);

    int n_trials = 5;

    /* 存储每条轨迹的 dR 元素和 dv, dp (每个采样点) */
    /* 为简单: 只存终态, 以及几个中间快照用于绘图 */
    double trail_R00[5][61];  /* 5 trials × 每 0.05s */
    double trail_dv_x[5][61];
    int n_rows = steps / 10;  /* 每 10 个采样输出一行 */

    for (int trial = 0; trial < n_trials; trial++) {
        lcg_state = (unsigned)(12345 + trial * 7777);

        imu_preint_t p;
        double ba[3] = {0,0,0}, bg[3] = {0,0,0};
        imu_preint_init(&p, sigma_a, sigma_g, 1e-6, 1e-6, ba, bg);

        int row = 0;
        for (int k = 0; k < steps; k++) {
            double gyr[3] = {randn() * sg_disc,
                             randn() * sg_disc,
                             omega_z + randn() * sg_disc};
            double acc[3] = {randn() * sa_disc,
                             randn() * sa_disc,
                             randn() * sa_disc};

            imu_preint_update(&p, acc, gyr, dt);

            if (k % 10 == 0 && row < 61) {
                double dR[9], dv[3], dp[3];
                imu_preint_get_delta(&p, dR, dv, dp);
                trail_R00[trial][row] = dR[0];
                trail_dv_x[trial][row] = dv[0];
                row++;
            }
        }
    }

    /* 输出: 真值 + 均值 ± 2σ 包络 */
    FILE *f = fopen("imu_vis_6_noisy.csv", "w");
    if (!f) { printf("  Cannot open output file\n"); return; }
    fprintf(f, "time,dR00_true,dR00_mean,dR00_lo,dR00_hi,"
            "dv_x_true,dv_x_mean,dv_x_lo,dv_x_hi\n");

    for (int row = 0; row < n_rows; row++) {
        double T = (row * 10 + 1) * dt;
        double theta = omega_z * T;
        double dR00_true = cos(theta);
        double dv_x_true = 0.0;  /* 纯旋转, 无平移 */

        /* 统计 5 条轨迹 */
        double sum_R = 0, sum_R2 = 0;
        double sum_v = 0, sum_v2 = 0;
        for (int t = 0; t < n_trials; t++) {
            double r = trail_R00[t][row];
            double v = trail_dv_x[t][row];
            sum_R += r;  sum_R2 += r*r;
            sum_v += v;  sum_v2 += v*v;
        }
        double mean_R = sum_R / n_trials;
        double std_R = sqrt(sum_R2/n_trials - mean_R*mean_R + 1e-30);
        double mean_v = sum_v / n_trials;
        double std_v = sqrt(sum_v2/n_trials - mean_v*mean_v + 1e-30);

        fprintf(f, "%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                T, dR00_true, mean_R, mean_R-2*std_R, mean_R+2*std_R,
                dv_x_true, mean_v, mean_v-2*std_v, mean_v+2*std_v);
    }
    fclose(f);

    /* 验证: 最后一行真值应在 2σ 包络内 */
    double T_last = ((n_rows - 1) * 10 + 1) * dt;
    double dR00_true = cos(T_last * omega_z);
    double sum_R = 0, sum_R2 = 0;
    for (int t = 0; t < n_trials; t++) {
        double r = trail_R00[t][n_rows-1];
        sum_R += r; sum_R2 += r*r;
    }
    double mean_R = sum_R / n_trials;
    double std_R = sqrt(sum_R2/n_trials - mean_R*mean_R + 1e-30);
    CHECK(fabs(mean_R - dR00_true) < 3*std_R + 1e-3, "noisy: R00 within 3σ");
    printf("  at t=%.3f: true=%.6f  mean=%.6f  std=%.6f\n",
           T_last, dR00_true, mean_R, std_R);
    printf("  wrote %d rows\n", n_rows);
}

/* ================================================================ */

int main(void)
{
    printf("=== IMU Preintegration Visualization Test ===\n\n");

    test1_pure_rotation();
    test2_constant_accel();
    test3_circular_motion();
    test4_covariance();
    test5_bias_correction();
    test6_noisy_rotation();

    printf("\n=== %d passed, %d failed ===\n", n_pass, n_fail);
    printf("CSV files written:\n");
    printf("  imu_vis_1_rotation.csv\n");
    printf("  imu_vis_2_accel.csv\n");
    printf("  imu_vis_3_circular.csv\n");
    printf("  imu_vis_4_covariance.csv\n");
    printf("  imu_vis_5_bias.csv\n");
    printf("  imu_vis_5_bias_snapshot.csv\n");
    printf("  imu_vis_6_noisy.csv\n");
    printf("\nRun: MATLAB plot_imu.m\n");

    return n_fail > 0 ? 1 : 0;
}
