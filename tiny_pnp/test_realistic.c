/*
 * 真实感 PnP 测试: 模拟智能车赛道场景
 * 
 * 12 个非共面地标 (8 地面 + 4 路牌)
 * 测试: 纯净 / σ=0.3px / σ=1.0px / 不同点数 / 纯平面
 */

#include "stdio.h"
#include "string.h"
#include "math.h"
#include "tiny_pnp.h"
#include "../tiny_linalg/tiny_linalg.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

static unsigned rng = 42;
static double rand_norm(void) {
    rng = rng * 1103515245U + 12345U;
    double r1 = (double)(rng & 0x7FFFFFFF) / 2147483648.0;
    rng = rng * 1103515245U + 12345U;
    double r2 = (double)(rng & 0x7FFFFFFF) / 2147483648.0;
    return sqrt(-2.0 * log(r1 + 1e-30)) * cos(2.0 * M_PI * r2);
}

static void project(const double *P, const double R[9], const double t[3],
                     double fx, double fy, double cx, double cy,
                     double *u, double *v)
{
    double x = R[0]*P[0] + R[1]*P[1] + R[2]*P[2] + t[0];
    double y = R[3]*P[0] + R[4]*P[1] + R[5]*P[2] + t[1];
    double z = R[6]*P[0] + R[7]*P[1] + R[8]*P[2] + t[2];
    *u = fx * x / z + cx;
    *v = fy * y / z + cy;
}

static void add_noise(double *pts2D, int n, double sigma) {
    for (int i = 0; i < 2*n; i++) pts2D[i] += sigma * rand_norm();
}

static double t_error(const double a[3], const double b[3]) {
    double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return sqrt(dx*dx+dy*dy+dz*dz);
}

int main(void)
{
    pnp_camera_t cam = {450.0, 450.0, 320.0, 240.0};

    /* 真值位姿: 任意合理旋转+平移, 保证所有点 Z_cam > 0 */
    double w_true[3] = {0.35, 0.1, 0.05};
    double R_true[9];
    mat33_exp_so3(w_true, R_true);
    double t_true[3] = {-0.3, 0.1, 2.5};

    /* 12 个地标 (世界坐标, 米): 8 地面 z=0 + 4 路牌 z=0.2~0.4 */
    double pts3D[36] = {
        2.0, 0.0, 0.0,   3.0, 0.3, 0.0,   3.0,-0.3, 0.0,
        1.5, 0.2, 0.0,   1.5,-0.2, 0.0,   2.5, 0.0, 0.0,
        4.0, 0.4, 0.0,   4.0,-0.4, 0.0,
        2.0, 0.0, 0.25,  3.0, 0.3, 0.40,
        3.0,-0.3, 0.40,  1.5, 0.0, 0.20,
    };
    int n = 12;

    /* 投影 */
    double pts2D_clean[24];
    for (int i = 0; i < n; i++)
        project(&pts3D[3*i], R_true, t_true,
                cam.fx, cam.fy, cam.cx, cam.cy,
                &pts2D_clean[2*i], &pts2D_clean[2*i+1]);

    printf("Camera: w=[%.2f,%.2f,%.2f] t=[%.3f,%.3f,%.3f]\n\n",
           w_true[0],w_true[1],w_true[2],t_true[0],t_true[1],t_true[2]);

    /* ==== A: Clean ==== */
    printf("=== A: Clean (no noise) ===\n");
    {
        pnp_result_t res;
        int ret = pnp_solve(pts3D, pts2D_clean, n, &cam, &res);
        CHECK(ret==0, "clean converged");
        CHECK(res.reproj_err < 1e-6, "clean reproj < 1e-6");
        CHECK(t_error(res.t, t_true) < 1e-6, "clean t_err < 1um");
        printf("  iter=%d reproj=%.2e t=[%.3f,%.3f,%.3f]\n",
               res.iterations, res.reproj_err, res.t[0],res.t[1],res.t[2]);
    }

    /* ==== B: σ=0.3px ==== */
    printf("\n=== B: σ=0.3 px (high quality) ===\n");
    {
        double p2d[24]; memcpy(p2d, pts2D_clean, sizeof(p2d));
        rng=42; add_noise(p2d, n, 0.3);
        pnp_result_t res;
        int ret = pnp_solve(pts3D, p2d, n, &cam, &res);
        CHECK(ret==0, "σ0.3 converged");
        CHECK(res.reproj_err < 0.6, "σ0.3 reproj < 0.6px");
        CHECK(t_error(res.t, t_true)*1000.0 < 5.0, "σ0.3 t_err < 5mm");
        printf("  iter=%d reproj=%.4fpx t_err=%.2fmm\n",
               res.iterations, res.reproj_err, t_error(res.t,t_true)*1000.0);
    }

    /* ==== C: σ=1.0px ==== */
    printf("\n=== C: σ=1.0 px (normal) ===\n");
    {
        double p2d[24]; memcpy(p2d, pts2D_clean, sizeof(p2d));
        rng=42; add_noise(p2d, n, 1.0);
        pnp_result_t res;
        int ret = pnp_solve(pts3D, p2d, n, &cam, &res);
        CHECK(ret==0, "σ1.0 converged");
        CHECK(res.reproj_err < 1.5, "σ1.0 reproj < 1.5px");
        CHECK(t_error(res.t, t_true)*1000.0 < 15.0, "σ1.0 t_err < 15mm");
        printf("  iter=%d reproj=%.4fpx t_err=%.2fmm\n",
               res.iterations, res.reproj_err, t_error(res.t,t_true)*1000.0);
    }

    /* ==== D: 不同点数 ==== */
    printf("\n=== D: Varying point count (σ=0.5px) ===\n");
    int counts[] = {5, 6, 8, 12};
    for (int k = 0; k < 4; k++) {
        int np = counts[k];
        double p3d[36], p2d[24];
        /* 混合选取: 地面点 + 高程点 (打破平面退化) */
        int n_ground = np - 2;   /* 至少 2 个高程点打破平面退化 */
        if (n_ground < 3) n_ground = 3;
        if (n_ground > 8) n_ground = 8;
        for (int i = 0; i < n_ground; i++) {
            p3d[3*i]=pts3D[3*i]; p3d[3*i+1]=pts3D[3*i+1]; p3d[3*i+2]=pts3D[3*i+2];
            p2d[2*i]=pts2D_clean[2*i]; p2d[2*i+1]=pts2D_clean[2*i+1];
        }
        for (int i = n_ground; i < np; i++) {
            int src = 8 + (i - n_ground);  /* 高程点从索引 8 开始 */
            p3d[3*i]=pts3D[3*src]; p3d[3*i+1]=pts3D[3*src+1]; p3d[3*i+2]=pts3D[3*src+2];
            p2d[2*i]=pts2D_clean[2*src]; p2d[2*i+1]=pts2D_clean[2*src+1];
        }
        rng=42; add_noise(p2d, np, 0.5);
        pnp_result_t res;
        int ret = pnp_solve(p3d, p2d, np, &cam, &res);
        double te = t_error(res.t, t_true);
        printf("  n=%2d: reproj=%.4fpx t_err=%.2fmm %s\n",
               np, res.reproj_err, te*1000.0, (ret==0&&te<0.02)?"OK":"FAIL");
        CHECK(ret==0 && te<0.02, "varying n OK");
    }

    /* ==== E: 纯平面 ==== */
    printf("\n=== E: Planar-only z=0 (σ=0.5px) ===\n");
    {
        double p3d[24], p2d[16]; int np=8;
        for (int i=0;i<np;i++) {
            p3d[3*i]=pts3D[3*i]; p3d[3*i+1]=pts3D[3*i+1]; p3d[3*i+2]=0.0;
            project(&p3d[3*i], R_true, t_true, cam.fx,cam.fy,cam.cx,cam.cy,
                    &p2d[2*i], &p2d[2*i+1]);
        }
        rng=42; add_noise(p2d, np, 0.5);
        pnp_result_t res;
        pnp_solve(p3d, p2d, np, &cam, &res);
        double te = t_error(res.t, t_true);
        printf("  reproj=%.4fpx t_err=%.2fmm z_est=%.3fm z_true=%.3fm\n",
               res.reproj_err, te*1000.0, fabs(res.t[2]), fabs(t_true[2]));
        printf("  → Planar target: Z/depth unreliable due to ambiguity\n");
        CHECK(res.reproj_err < 5.0, "planar reproj bounded");
    }

    printf("\n%d passed, %d failed\n\n", n_pass, n_fail);
    printf("=== Real-image workflow ===\n");
    printf("  1. Detect landmark corners → pixel coords (u,v)\n");
    printf("  2. Pre-survey landmark 3D positions in world frame\n");
    printf("  3. pnp_solve(pts3D, pts2D, n, &cam, &res);\n");
    printf("  4. P_cam = res.R * P_world + res.t\n");

    return n_fail>0?1:0;
}
