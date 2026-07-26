#include "stdio.h"
#include "math.h"
#include "tiny_ekf.h"
#include "../tiny_linalg/tiny_linalg.h"

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL [%d]: %s\n", __LINE__, msg); } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(fabs((a)-(b)) < (tol), msg)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * 测试 1: 1D 匀速运动 — 纯预测
 * 验证: 预测步正确传播状态和协方差
 * ================================================================ */

static void cv_transition(const double *x, const double *u, double dt,
                           double *x_pred, double *F, int n)
{
    (void)u; (void)n;
    x_pred[0] = x[0] + x[1] * dt;   /* pos = pos + vel*dt */
    x_pred[1] = x[1];                /* vel unchanged       */
    /* F = [[1, dt], [0, 1]] */
    F[0] = 1.0;  F[1] = dt;
    F[2] = 0.0;  F[3] = 1.0;
}

static void test_predict_only(void)
{
    printf("1D constant velocity (predict only) ...\n");

    ekf_t ekf;
    double x0[] = {0.0, 1.0};
    double P0[] = {1.0, 0.0, 0.0, 0.1};  /* pos uncertainty 1, vel 0.1 */
    ekf_init(&ekf, 2, x0, P0);

    double Q[] = {0.01, 0.0, 0.0, 0.01};

    for (int i = 0; i < 10; i++)
        ekf_predict(&ekf, cv_transition, NULL, 0.1, Q);

    /* pos 应增长: 0 + 1*10*0.1 = 1.0 */
    CHECK_NEAR(ekf.x[0], 1.0, 1e-10, "predict pos=1.0");
    CHECK_NEAR(ekf.x[1], 1.0, 1e-10, "predict vel=1.0");
    /* 协方差对角线应增大 (过程噪声累积) */
    CHECK(ekf.P[0] > P0[0], "P(pos,pos) grew");
    CHECK(ekf.P[3] > P0[3], "P(vel,vel) grew");
}

/* ================================================================
 * 测试 2: 1D 位置观测 — 预测+更新
 * 验证: 更新步减小协方差, 状态收敛到真值
 * ================================================================ */

static void pos_observation(const double *x, double *z_pred, double *H,
                             int m, int n)
{
    (void)n;
    z_pred[0] = x[0];   /* 直接观测位置 */
    H[0] = 1.0;          /* ∂z/∂pos = 1 */
    H[1] = 0.0;          /* ∂z/∂vel = 0 */
    (void)m;
}

static void test_position_update(void)
{
    printf("1D position update ...\n");

    ekf_t ekf;
    double x0[] = {0.0, 0.0};           /* 初始不知道速度 */
    double P0[] = {1.0, 0.0, 0.0, 10.0}; /* vel 不确定性大 */
    ekf_init(&ekf, 2, x0, P0);

    double Q[] = {0.001, 0.0, 0.0, 0.001};
    double R[] = {0.01};                 /* 位置测量噪声 */

    double true_pos = 0.0;
    double true_vel = 1.0;
    double dt = 0.1;

    for (int i = 0; i < 50; i++) {
        /* 预测 */
        ekf_predict(&ekf, cv_transition, NULL, dt, Q);
        true_pos += true_vel * dt;

        /* 无噪声观测 (确定性的) */
        double z[] = {true_pos};
        ekf_update(&ekf, pos_observation, z, R, 1);
    }

    /* 50 步后 pos≈5.0, vel 应已收敛到 ≈1.0 */
    CHECK_NEAR(ekf.x[0], true_pos, 1e-3, "update pos≈true");
    CHECK_NEAR(ekf.x[1], true_vel, 1e-2, "update vel≈1.0");
    /* 协方差应显著缩小 */
    CHECK(ekf.P[0] < P0[0] * 0.5, "P(pos) shrank");
    CHECK(ekf.P[3] < P0[3] * 0.5, "P(vel) shrank");
    printf("  pos=%.4f, vel=%.4f, P_diag=[%.4f, %.4f]\n",
           ekf.x[0], ekf.x[1], ekf.P[0], ekf.P[3]);
}

/* ================================================================
 * 测试 3: 2D 机器人 range+bearing 定位
 * 状态 [x, y, θ], 控制 v=1, ω=0.1, 2个地标 → m=4
 * ================================================================ */

/* 地标位置 */
static double lm0[2] = {5.0, 0.0};
static double lm1[2] = {0.0, 5.0};

static void robot_transition(const double *x, const double *u, double dt,
                              double *x_pred, double *F, int n)
{
    (void)n;
    double v  = u ? u[0] : 1.0;
    double w  = u ? u[1] : 0.1;
    double th = x[2];

    x_pred[0] = x[0] + v * cos(th) * dt;
    x_pred[1] = x[1] + v * sin(th) * dt;
    x_pred[2] = x[2] + w * dt;

    /* F = ∂f/∂x (3×3, 行优先) */
    F[0] = 1.0;  F[1] = 0.0;  F[2] = -v * sin(th) * dt;
    F[3] = 0.0;  F[4] = 1.0;  F[5] =  v * cos(th) * dt;
    F[6] = 0.0;  F[7] = 0.0;  F[8] = 1.0;
}

static void range_bearing_obs(const double *x, double *z_pred, double *H,
                               int m, int n)
{
    (void)m; (void)n;
    double dx0 = lm0[0] - x[0], dy0 = lm0[1] - x[1];
    double dx1 = lm1[0] - x[0], dy1 = lm1[1] - x[1];
    double r0 = sqrt(dx0*dx0 + dy0*dy0);
    double r1 = sqrt(dx1*dx1 + dy1*dy1);
    double b0 = atan2(dy0, dx0) - x[2];
    double b1 = atan2(dy1, dx1) - x[2];

    z_pred[0] = r0;  z_pred[1] = b0;
    z_pred[2] = r1;  z_pred[3] = b1;

    /* H = ∂(r0,b0,r1,b1)/∂(x,y,θ), 4×3 行优先 */
    double ir0 = 1.0 / (r0 > 1e-10 ? r0 : 1e-10);
    double ir1 = 1.0 / (r1 > 1e-10 ? r1 : 1e-10);
    double r0sq = ir0 * ir0;
    double r1sq = ir1 * ir1;

    /* row 0: ∂r0 */
    H[0] = -dx0 * ir0;    H[1] = -dy0 * ir0;    H[2] = 0.0;
    /* row 1: ∂b0 */
    H[3] =  dy0 * r0sq;   H[4] = -dx0 * r0sq;   H[5] = -1.0;
    /* row 2: ∂r1 */
    H[6] = -dx1 * ir1;    H[7] = -dy1 * ir1;    H[8] = 0.0;
    /* row 3: ∂b1 */
    H[9] =  dy1 * r1sq;   H[10]= -dx1 * r1sq;   H[11]= -1.0;
}

static void test_robot_localization(void)
{
    printf("2D robot range+bearing localization ...\n");

    ekf_t ekf;
    double x0[] = {0.1, 0.1, 0.05};  /* 偏离真值 (0,0,0) */
    double P0[9] = {0.5,0,0, 0,0.5,0, 0,0,0.1};
    ekf_init(&ekf, 3, x0, P0);

    /* 过程噪声 */
    double Q[] = {0.01,0,0, 0,0.01,0, 0,0,0.001};
    /* 观测噪声: range σ=0.1, bearing σ=0.02 rad */
    double R[] = {0.01,0,0,0, 0,0.0004,0,0, 0,0,0.01,0, 0,0,0,0.0004};

    double dt = 0.1;
    double true_x = 0.0, true_y = 0.0, true_th = 0.0;
    double v = 1.0, w = 0.1;

    for (int i = 0; i < 100; i++) {
        double u[] = {v, w};
        ekf_predict(&ekf, robot_transition, u, dt, Q);

        /* 真值更新 */
        true_x  += v * cos(true_th) * dt;
        true_y  += v * sin(true_th) * dt;
        true_th += w * dt;

        /* 计算观测 (真值处的) */
        double dx0 = lm0[0] - true_x, dy0 = lm0[1] - true_y;
        double dx1 = lm1[0] - true_x, dy1 = lm1[1] - true_y;
        double z[4];
        z[0] = sqrt(dx0*dx0 + dy0*dy0);
        z[1] = atan2(dy0, dx0) - true_th;
        z[2] = sqrt(dx1*dx1 + dy1*dy1);
        z[3] = atan2(dy1, dx1) - true_th;

        ekf_update(&ekf, range_bearing_obs, z, R, 4);
    }

    /* 100 步后应接近真值 */
    CHECK_NEAR(ekf.x[0], true_x,  0.1, "robot x≈true");
    CHECK_NEAR(ekf.x[1], true_y,  0.1, "robot y≈true");
    CHECK_NEAR(ekf.x[2], true_th, 0.05,"robot θ≈true");
    /* 协方差应缩小 */
    CHECK(ekf.P[0] < P0[0], "robot P(x) shrank");
    CHECK(ekf.P[4] < P0[4], "robot P(y) shrank");
    CHECK(ekf.P[8] < P0[8], "robot P(θ) shrank");
    printf("  est=[%.4f, %.4f, %.4f] true=[%.4f, %.4f, %.4f]\n",
           ekf.x[0], ekf.x[1], ekf.x[2], true_x, true_y, true_th);
}

/* ================================================================
 * 测试 4: Joseph 形式验证 — P 保持对称性
 * ================================================================ */

static void test_joseph_symmetry(void)
{
    printf("Joseph form symmetry ...\n");

    ekf_t ekf;
    double x0[] = {0.0, 0.0};
    /* 非对角的初始协方差 */
    double P0[] = {1.0, 0.3, 0.3, 0.5};
    ekf_init(&ekf, 2, x0, P0);

    double Q[] = {0.01, 0.0, 0.0, 0.01};
    double R[] = {0.04};
    double z[]  = {1.0};
    double dt = 0.1;

    /* 多轮预测+更新 */
    for (int i = 0; i < 20; i++) {
        ekf_predict(&ekf, cv_transition, NULL, dt, Q);
        ekf_update(&ekf, pos_observation, z, R, 1);
    }

    /* P 应对称: P[0,1] == P[1,0] */
    double asym = fabs(ekf.P[1] - ekf.P[2]);  /* P[0,1] vs P[1,0] */
    CHECK(asym < 1e-14, "P is symmetric");

    /* P 应为正定: 对角元 > 0 */
    CHECK(ekf.P[0] > 0.0, "P[0,0] > 0");
    CHECK(ekf.P[3] > 0.0, "P[1,1] > 0");
    /* 行列式 > 0 (2×2: a*d - b*c > 0) */
    double det = ekf.P[0] * ekf.P[3] - ekf.P[1] * ekf.P[2];
    CHECK(det > 0.0, "P det > 0 (pos-def)");
    printf("  P = [[%.6f, %.6f], [%.6f, %.6f]], det=%.6e\n",
           ekf.P[0], ekf.P[1], ekf.P[2], ekf.P[3], det);
}

/* ================================================================
 * 测试 5: 边界情况
 * ================================================================ */

static void test_edge_cases(void)
{
    printf("edge cases ...\n");

    ekf_t ekf;
    double x0[] = {0.0, 0.0};
    ekf_init(&ekf, 2, x0, NULL);  /* NULL P0 → 默认大对角 */
    CHECK(ekf.n == 2, "init n=2");
    CHECK(ekf.P[0] > 100.0, "default P0 large diag");
    CHECK(ekf.P[1] == 0.0, "default P0 off-diag=0");

    /* m=0 更新应直接返回 */
    int ret = ekf_update(&ekf, NULL, NULL, NULL, 0);
    CHECK(ret == 0, "m=0 returns 0");
}

/* ================================================================ */

int main(void)
{
    test_predict_only();
    test_position_update();
    test_robot_localization();
    test_joseph_symmetry();
    test_edge_cases();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
