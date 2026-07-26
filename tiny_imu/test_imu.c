/*
 * @Author: garygoo
 * @Date: 2026-07-26
 * @Description: tiny_imu 测试套件 — IMU 预积分
 *   编译: gcc -std=c99 -Wall -Wextra -O2 -o test_imu
 *         test_imu.c tiny_imu.c ../tiny_linalg/tiny_linalg.c -lm
 */

#include "stdio.h"
#include "math.h"
#include "tiny_imu.h"
#include "../tiny_linalg/tiny_linalg.h"

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL [%d]: %s\n", __LINE__, msg); } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(fabs((a)-(b)) < (tol), msg)

/* ================================================================
 * 测试 1: 初始化与重置
 * ================================================================ */
static void test_init_reset(void)
{
    printf("init / reset ...\n");

    imu_preint_t p;
    double ba[3] = {0.01, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.001};

    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* 检查初始化状态 */
    CHECK_NEAR(p.dR[0], 1.0, 1e-15, "dR[0]=1");
    CHECK_NEAR(p.dR[4], 1.0, 1e-15, "dR[4]=1");
    CHECK_NEAR(p.dR[8], 1.0, 1e-15, "dR[8]=1");
    CHECK_NEAR(p.dR[1], 0.0, 1e-15, "dR[1]=0");
    CHECK_NEAR(p.dv[0], 0.0, 1e-15, "dv[0]=0");
    CHECK_NEAR(p.dp[0], 0.0, 1e-15, "dp[0]=0");
    CHECK_NEAR(p.dt,   0.0, 1e-15, "dt=0");
    CHECK_NEAR(p.cov[0], 0.0, 1e-15, "cov[0]=0");
    CHECK_NEAR(p.J_dR_bg[0], 0.0, 1e-15, "J_dR_bg[0]=0");
    CHECK_NEAR(p.sigma_a, 0.01, 1e-15, "sigma_a");
    CHECK_NEAR(p.ba[0], 0.01, 1e-15, "ba[0]");

    /* 加入一次采样后重置 */
    double acc[3] = {1.0, 0.0, 0.0};
    double gyr[3] = {0.0, 0.0, 0.1};
    imu_preint_update(&p, acc, gyr, 0.01);
    CHECK(p.dt > 0.0, "dt>0 after update");

    double ba2[3] = {0.02, 0.0, 0.0};
    double bg2[3] = {0.0, 0.0, 0.002};
    imu_preint_reset(&p, ba2, bg2);

    CHECK_NEAR(p.dR[0], 1.0, 1e-15, "dR[0]=1 after reset");
    CHECK_NEAR(p.dv[0], 0.0, 1e-15, "dv[0]=0 after reset");
    CHECK_NEAR(p.dp[0], 0.0, 1e-15, "dp[0]=0 after reset");
    CHECK_NEAR(p.dt,   0.0, 1e-15, "dt=0 after reset");
    CHECK_NEAR(p.cov[0], 0.0, 1e-15, "cov[0]=0 after reset");
    CHECK_NEAR(p.J_dR_bg[0], 0.0, 1e-15, "J_dR_bg[0]=0 after reset");
    CHECK_NEAR(p.ba[0], 0.02, 1e-15, "ba[0] updated");
}

/* ================================================================
 * 测试 2: 静止 IMU — 零运动
 * ================================================================ */
static void test_zero_motion(void)
{
    printf("zero motion ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* 静止 IMU: 无加速度, 无角速度 */
    double acc[3] = {0.0, 0.0, 0.0};
    double gyr[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < 10; i++)
        imu_preint_update(&p, acc, gyr, 0.005);

    /* 无运动, 预积分量应保持单位/零 */
    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);

    CHECK_NEAR(dR[0], 1.0, 1e-15, "zero: dR=I[0]");
    CHECK_NEAR(dR[4], 1.0, 1e-15, "zero: dR=I[4]");
    CHECK_NEAR(dR[8], 1.0, 1e-15, "zero: dR=I[8]");
    CHECK_NEAR(dR[1], 0.0, 1e-15, "zero: dR off-diag");
    CHECK_NEAR(dv[0], 0.0, 1e-12, "zero: dv[0]=0");
    CHECK_NEAR(dv[1], 0.0, 1e-12, "zero: dv[1]=0");
    CHECK_NEAR(dp[0], 0.0, 1e-12, "zero: dp[0]=0");
    CHECK(p.dt > 0.049 && p.dt < 0.051, "zero: dt≈0.05");

    /* 协方差应增长 (噪声在累积) */
    CHECK(p.cov[0] > 0.0, "zero: cov grows");
}

/* ================================================================
 * 测试 3: 匀速直线运动 (纯平移, 无旋转)
 * ================================================================ */
static void test_constant_velocity(void)
{
    printf("constant velocity (pure translation) ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* 机体 X 方向恒定加速度 = 已补偿重力后的比力
     * 实际飞行器水平前飞: acc_x ≈ 前进推力/质量
     * 此处简化为常量 a_body = [2, 0, 0] */
    double acc[3] = {2.0, 0.0, 0.0};
    double gyr[3] = {0.0, 0.0, 0.0};

    double dt = 0.005;
    int n = 20;  /* 0.1s 总计 */
    for (int i = 0; i < n; i++)
        imu_preint_update(&p, acc, gyr, dt);

    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);

    /* 无旋转: dR=I */
    CHECK_NEAR(dR[0], 1.0, 1e-14, "const_v: dR=I");

    /* Δv = a·T = 2.0 * 0.1 = 0.2 m/s */
    CHECK_NEAR(dv[0], 0.2, 1e-10, "const_v: dv[0]=0.2");
    CHECK_NEAR(dv[1], 0.0, 1e-14, "const_v: dv[1]=0");

    /* Δp = 0.5*a*T² = 0.5*2.0*0.01 = 0.01 m */
    CHECK_NEAR(dp[0], 0.01, 1e-10, "const_v: dp[0]=0.01");
    CHECK_NEAR(dp[1], 0.0, 1e-14, "const_v: dp[1]=0");
}

/* ================================================================
 * 测试 4: 纯旋转 (恒定角速度绕 Z 轴)
 * ================================================================ */
static void test_pure_rotation(void)
{
    printf("pure rotation (Z axis) ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* ω_z = 1.0 rad/s, a=0 */
    double acc[3] = {0.0, 0.0, 0.0};
    double gyr[3] = {0.0, 0.0, 1.0};

    double dt = 0.01;
    int n = 100;  /* 1.0s → 总转角 = 1.0 rad ≈ 57.3° */
    for (int i = 0; i < n; i++)
        imu_preint_update(&p, acc, gyr, dt);

    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);

    /* 绕 Z 轴旋转 θ=1.0 rad:
     * R = [[cosθ, -sinθ, 0],
     *      [sinθ,  cosθ, 0],
     *      [0,     0,     1]] */
    double c = cos(1.0), s = sin(1.0);
    CHECK_NEAR(dR[0],  c, 1e-10, "rot: R00=cos(1)");
    CHECK_NEAR(dR[1], -s, 1e-10, "rot: R01=-sin(1)");
    CHECK_NEAR(dR[3],  s, 1e-10, "rot: R10=sin(1)");
    CHECK_NEAR(dR[4],  c, 1e-10, "rot: R11=cos(1)");
    CHECK_NEAR(dR[8], 1.0, 1e-10, "rot: R22=1");
    CHECK_NEAR(dR[2], 0.0, 1e-14, "rot: R02=0");

    /* 无平移: dv=0, dp=0 */
    CHECK_NEAR(dv[0], 0.0, 1e-12, "rot: dv=0");
    CHECK_NEAR(dp[0], 0.0, 1e-12, "rot: dp=0");
}

/* ================================================================
 * 测试 5: 常加速度 (机体 X 轴, 无旋转) — 抛物线轨迹
 * ================================================================ */
static void test_constant_accel(void)
{
    printf("constant acceleration (parabolic) ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* X 方向恒定加速度 a=3.0 m/s², 机体始终朝同一方向 */
    double acc[3] = {3.0, 0.0, 0.0};
    double gyr[3] = {0.0, 0.0, 0.0};

    double dt = 0.005;
    int n = 40;  /* T = 0.2s */
    for (int i = 0; i < n; i++)
        imu_preint_update(&p, acc, gyr, dt);

    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p, dR, dv, dp);

    double T = n * dt;  /* 0.2s */
    /* Δv = a·T = 3.0 * 0.2 = 0.6 */
    CHECK_NEAR(dv[0], 3.0 * T, 1e-10, "accel: dv[0]");
    /* Δp = 0.5*a*T² = 0.5*3*0.04 = 0.06 */
    CHECK_NEAR(dp[0], 0.5 * 3.0 * T * T, 1e-10, "accel: dp[0]");
}

/* ================================================================
 * 测试 6: 带偏置的积分 + 偏置修正
 * ================================================================ */
static void test_bias_correction(void)
{
    printf("bias correction ...\n");

    /* 用零偏置做参考积分 */
    imu_preint_t p_ref;
    double ba0[3] = {0.0, 0.0, 0.0};
    double bg0[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p_ref, 1e-6, 1e-6, 1e-8, 1e-8, ba0, bg0);

    /* 用带偏置的估计做积分 */
    imu_preint_t p;
    double ba[3] = {0.05, -0.02, 0.01};  /* 故意设错偏置 */
    double bg[3] = {0.001, 0.0, -0.002};
    imu_preint_init(&p, 1e-6, 1e-6, 1e-8, 1e-8, ba, bg);

    /* 真实 IMU 数据 (匀速转+加速) */
    double dt = 0.005;
    for (int k = 0; k < 60; k++) {  /* 0.3s */
        double t = k * dt;
        double acc[3] = {2.0*sin(t), 1.0, 0.0};
        double gyr[3] = {0.0, 0.0, 0.5};

        /* 参考: 用真实偏置=零积分 */
        imu_preint_update(&p_ref, acc, gyr, dt);
        /* 带偏置: 用错误偏置积分 */
        imu_preint_update(&p, acc, gyr, dt);
    }

    /* 偏置修正: 把 p 的偏置修正为真实值(0), 结果应接近 p_ref */
    imu_preint_bias_correct(&p, ba0, bg0);

    double dR_ref[9], dv_ref[3], dp_ref[3];
    double dR[9], dv[3], dp[3];
    imu_preint_get_delta(&p_ref, dR_ref, dv_ref, dp_ref);
    imu_preint_get_delta(&p, dR, dv, dp);

    /* 一阶修正后应接近参考值 */
    CHECK_NEAR(dR[0], dR_ref[0], 1e-6, "bias: dR00 match");
    CHECK_NEAR(dR[4], dR_ref[4], 1e-6, "bias: dR11 match");
    CHECK_NEAR(dv[0], dv_ref[0], 1e-4, "bias: dv[0] match");
    CHECK_NEAR(dv[1], dv_ref[1], 1e-4, "bias: dv[1] match");
    CHECK_NEAR(dp[0], dp_ref[0], 1e-4, "bias: dp[0] match");
}

/* ================================================================
 * 测试 7: 协方差增长 (不确定性随时间增加)
 * ================================================================ */
static void test_covariance_growth(void)
{
    printf("covariance growth ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    /* 较大噪声便于观察协方差增长 */
    imu_preint_init(&p, 0.1, 0.05, 1e-4, 1e-4, ba, bg);

    double acc[3] = {1.0, 0.5, 0.0};
    double gyr[3] = {0.1, 0.0, 0.2};

    /* 获取初始协方差 trace */
    double trace_before = 0.0;
    for (int i = 0; i < 9; i++) trace_before += p.cov[i * 9 + i];

    /* 多次采样 */
    for (int k = 0; k < 50; k++)
        imu_preint_update(&p, acc, gyr, 0.01);

    double trace_after = 0.0;
    for (int i = 0; i < 9; i++) trace_after += p.cov[i * 9 + i];

    CHECK(trace_after > trace_before, "cov: trace increased");
    CHECK(p.cov[0] > 0.0, "cov: δθ variance >0");
    CHECK(p.cov[3*9+3] > 0.0, "cov: δv variance >0");
    CHECK(p.cov[6*9+6] > 0.0, "cov: δp variance >0");
}

/* ================================================================
 * 测试 8: dt=0 边界检查
 * ================================================================ */
static void test_edge_cases(void)
{
    printf("edge cases ...\n");

    imu_preint_t p;
    double ba[3] = {0.0, 0.0, 0.0};
    double bg[3] = {0.0, 0.0, 0.0};
    imu_preint_init(&p, 0.01, 0.001, 1e-5, 1e-6, ba, bg);

    /* dt=0 应该不做任何改变 */
    double acc[3] = {9.8, 0.0, 0.0};
    double gyr[3] = {0.0, 1.0, 0.0};
    imu_preint_update(&p, acc, gyr, 0.0);

    CHECK_NEAR(p.dt, 0.0, 1e-15, "edge: dt still 0");
    CHECK_NEAR(p.dR[0], 1.0, 1e-15, "edge: dR still I");

    /* 负 dt 也应被忽略 */
    imu_preint_update(&p, acc, gyr, -0.01);
    CHECK_NEAR(p.dt, 0.0, 1e-15, "edge: neg dt ignored");
}

/* ================================================================ */

int main(void)
{
    test_init_reset();
    test_zero_motion();
    test_constant_velocity();
    test_pure_rotation();
    test_constant_accel();
    test_bias_correction();
    test_covariance_growth();
    test_edge_cases();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
