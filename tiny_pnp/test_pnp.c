#include "stdio.h"
#include "math.h"
#include "tiny_pnp.h"
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
 * 辅助: 用 R,t 将 3D 点投影到 2D
 * ================================================================ */

static void project_points(const double *pts3D, int n,
                            const double R[9], const double t[3],
                            double fx, double fy, double cx, double cy,
                            double *pts2D)
{
    for (int i = 0; i < n; i++) {
        const double *P = &pts3D[i*3];
        double x = R[0]*P[0] + R[1]*P[1] + R[2]*P[2] + t[0];
        double y = R[3]*P[0] + R[4]*P[1] + R[5]*P[2] + t[1];
        double z = R[6]*P[0] + R[7]*P[1] + R[8]*P[2] + t[2];
        pts2D[2*i]   = fx * x / z + cx;
        pts2D[2*i+1] = fy * y / z + cy;
    }
}

/* 将旋转向量转为旋转矩阵 */
static void rot_vec_to_mat(const double w[3], double R[9])
{
    mat33_exp_so3(w, R);
}

/* ================================================================
 * 测试 1: 正面相机, 4 个共面点
 * 相机在 (0,0,5), 看 z=0 平面上的方形
 * ================================================================ */

static void test_frontal_pose(void)
{
    printf("frontal pose (4 coplanar points) ...\n");

    double pts3D[12] = {
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0
    };
    double R_true[9] = {1,0,0, 0,1,0, 0,0,1};
    double t_true[3] = {0, 0, 5};

    pnp_camera_t cam = {800, 800, 320, 240};
    double pts2D[8];
    project_points(pts3D, 4, R_true, t_true, cam.fx, cam.fy, cam.cx, cam.cy, pts2D);

    pnp_result_t res;
    int ret = pnp_solve(pts3D, pts2D, 4, &cam, &res);

    CHECK(ret == 0, "frontal: converged");
    /* R 应 ≈ I */
    CHECK_NEAR(res.R[0], 1.0, 1e-4, "frontal: R[0]≈1");
    CHECK_NEAR(res.R[4], 1.0, 1e-4, "frontal: R[4]≈1");
    CHECK_NEAR(res.R[8], 1.0, 1e-4, "frontal: R[8]≈1");
    /* t 应 ≈ [0, 0, 5] */
    CHECK_NEAR(res.t[0], 0.0, 1e-4, "frontal: tx≈0");
    CHECK_NEAR(res.t[1], 0.0, 1e-4, "frontal: ty≈0");
    CHECK_NEAR(res.t[2], 5.0, 1e-3, "frontal: tz≈5");
    CHECK(res.reproj_err < 1e-6, "frontal: low reproj err");

    printf("  t=[%.4f, %.4f, %.4f] reproj=%.2e\n",
           res.t[0], res.t[1], res.t[2], res.reproj_err);
}

/* ================================================================
 * 测试 2: 任意位姿, 12 个随机 3D 点
 * 用真值 R,t 投影 → pnp_solve 恢复 → 验证精度
 * ================================================================ */

static void test_arbitrary_pose(void)
{
    printf("arbitrary pose (12 random points) ...\n");

    /* 真值位姿: 绕 (0.5, -0.3, 0.8) 旋转 + 平移 */
    double w_true[3] = {0.5, -0.3, 0.8};
    double R_true[9];
    rot_vec_to_mat(w_true, R_true);
    double t_true[3] = {0.5, -0.2, 4.0};

    pnp_camera_t cam = {600, 600, 300, 200};

    /* 12 个随机 3D 点 (z=0..3 范围) */
    double pts3D[36] = {
        0.1, 0.2, 0.5,   1.5, 0.3, 1.2,   2.1,-0.5, 0.8,
       -0.8, 1.3, 2.1,   0.5,-1.2, 1.5,  -1.5,-0.3, 2.8,
        0.0, 0.0, 3.0,   2.0, 1.5, 0.3,  -2.0, 2.0, 1.0,
        1.0,-1.0, 2.5,  -1.0,-2.0, 0.1,   3.0, 0.0, 1.8
    };
    int n = 12;

    double pts2D[24];
    project_points(pts3D, n, R_true, t_true,
                   cam.fx, cam.fy, cam.cx, cam.cy, pts2D);

    pnp_result_t res;
    int ret = pnp_solve(pts3D, pts2D, n, &cam, &res);

    CHECK(ret == 0, "arbitrary: converged");

    /* 验证: 用恢复的 R,t 重投影, 误差应极小 */
    double pts2D_chk[24];
    project_points(pts3D, n, res.R, res.t,
                   cam.fx, cam.fy, cam.cx, cam.cy, pts2D_chk);
    double max_err = 0.0;
    for (int i = 0; i < 2*n; i++) {
        double e = fabs(pts2D_chk[i] - pts2D[i]);
        if (e > max_err) max_err = e;
    }
    CHECK(max_err < 1e-4, "arbitrary: reprojection accurate");
    CHECK(res.reproj_err < 1e-4, "arbitrary: low reproj err");

    /* 验证 t 恢复 */
    CHECK_NEAR(res.t[0], t_true[0], 1e-3, "arbitrary: tx");
    CHECK_NEAR(res.t[1], t_true[1], 1e-3, "arbitrary: ty");
    CHECK_NEAR(res.t[2], t_true[2], 1e-3, "arbitrary: tz");

    printf("  max_reproj=%.2e, t_error=[%.2e, %.2e, %.2e], lm_iter=%d\n",
           max_err, res.t[0]-t_true[0], res.t[1]-t_true[1],
           res.t[2]-t_true[2], res.iterations);
}

/* ================================================================
 * 测试 3: 带噪声 — 验证 LM 精化能改善 EPnP 线性解
 * ================================================================ */

static void test_noisy_data(void)
{
    printf("noisy data (Gaussian noise σ=1px) ...\n");

    double w_true[3] = {0.3, 0.1, -0.2};
    double R_true[9];
    rot_vec_to_mat(w_true, R_true);
    double t_true[3] = {0.2, -0.1, 3.5};

    pnp_camera_t cam = {500, 500, 250, 250};
    int n = 8;

    double pts3D[24] = {
       -0.5,-0.5, 0.0,   0.5,-0.5, 0.0,   0.5, 0.5, 0.0,  -0.5, 0.5, 0.0,
       -0.8, 0.0, 1.5,   0.0, 0.8, 1.5,   0.8, 0.0, 1.5,   0.0,-0.8, 1.5
    };

    double pts2D[16];
    project_points(pts3D, n, R_true, t_true,
                   cam.fx, cam.fy, cam.cx, cam.cy, pts2D);

    /* 加高斯噪声 σ=1.0 像素 */
    double pts2D_noisy[16];
    /* 伪随机种子 (固定以复现) */
    unsigned seed = 12345;
    for (int i = 0; i < 2*n; i++) {
        seed = seed * 1103515245U + 12345U;
        double r1 = (double)(seed & 0x7FFFFFFF) / 2147483648.0;
        seed = seed * 1103515245U + 12345U;
        double r2 = (double)(seed & 0x7FFFFFFF) / 2147483648.0;
        /* Box-Muller: N(0,1) */
        double g = sqrt(-2.0 * log(r1 + 1e-30)) * cos(2.0 * M_PI * r2);
        pts2D_noisy[i] = pts2D[i] + 1.0 * g;
    }

    pnp_result_t res;
    int ret = pnp_solve(pts3D, pts2D_noisy, n, &cam, &res);

    CHECK(ret == 0, "noisy: converged");
    /* 重投影误差应显著小于噪声水平 (LM 精化后) */
    CHECK(res.reproj_err < 2.0, "noisy: reproj RMS < 2px");

    /* R,t 应大致接近真值 */
    double t_err = sqrt((res.t[0]-t_true[0])*(res.t[0]-t_true[0])
                      + (res.t[1]-t_true[1])*(res.t[1]-t_true[1])
                      + (res.t[2]-t_true[2])*(res.t[2]-t_true[2]));
    CHECK(t_err < 0.5, "noisy: t error < 0.5");

    printf("  reproj_RMS=%.4f, t_error=%.4f, lm_iter=%d\n",
           res.reproj_err, t_err, res.iterations);
}

/* ================================================================
 * 测试 4: 边界检查
 * ================================================================ */

static void test_edge_cases(void)
{
    printf("edge cases ...\n");

    pnp_result_t res;
    double pts3D[12], pts2D[8];
    pnp_camera_t cam = {800, 800, 320, 240};

    /* n < 4 */
    CHECK(pnp_solve(pts3D, pts2D, 3, &cam, &res) == -1, "n=3 → -1");

    /* NULL cam */
    CHECK(pnp_solve(pts3D, pts2D, 4, NULL, &res) == -1, "cam=NULL → -1");

    /* NULL result */
    CHECK(pnp_solve(pts3D, pts2D, 4, &cam, NULL) == -1, "result=NULL → -1");
}

/* ================================================================ */

int main(void)
{
    test_frontal_pose();
    test_arbitrary_pose();
    test_noisy_data();
    test_edge_cases();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
