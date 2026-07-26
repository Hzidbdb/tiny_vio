/*
 * @Author: garygoo
 * @Date: 2026-07-25
 * @Description: EPnP 位姿解算 + Horn 绝对定向 + LM 重投影精化
 *   依赖: tiny_linalg (mat33_symeig, mat_power_inv, mat33_svd, mat_solve,
 *         mat_mul, mat_transpose, mat33_exp_so3, mat33_log_so3, ...)
 *         tiny_opt   (lm_solve)
 */

#include "string.h"
#include "math.h"
#include "tiny_pnp.h"
#include "../tiny_linalg/tiny_linalg.h"
#include "../tiny_opt/tiny_opt.h"

/* ========== 内部常量 ========== */

#define MAX_N 4           /* 零空间最大维数 */

/* ========== 步骤 1: 选择 4 个控制点 ========== */

static void select_control_points(const double *pts3D, int n,
                                   double ctrl[12])
{
    /* 质心 */
    double centroid[3] = {0, 0, 0};
    for (int i = 0; i < n; i++) {
        centroid[0] += pts3D[i*3 + 0];
        centroid[1] += pts3D[i*3 + 1];
        centroid[2] += pts3D[i*3 + 2];
    }
    double inv_n = 1.0 / n;
    centroid[0] *= inv_n; centroid[1] *= inv_n; centroid[2] *= inv_n;

    /* 协方差矩阵 (3×3) */
    double cov[9] = {0};
    for (int i = 0; i < n; i++) {
        double dx = pts3D[i*3+0] - centroid[0];
        double dy = pts3D[i*3+1] - centroid[1];
        double dz = pts3D[i*3+2] - centroid[2];
        cov[0] += dx*dx;  cov[1] += dx*dy;  cov[2] += dx*dz;
        cov[3] += dy*dx;  cov[4] += dy*dy;  cov[5] += dy*dz;
        cov[6] += dz*dx;  cov[7] += dz*dy;  cov[8] += dz*dz;
    }
    for (int i = 0; i < 9; i++) cov[i] *= inv_n;

    /* 特征分解 */
    double lambda[3], V[9];
    mat33_symeig(cov, lambda, V);

    /* 4 个控制点: c1=质心, c2..c4 = 质心 + √λ * v */
    for (int j = 0; j < 3; j++) ctrl[j] = centroid[j];   /* c1 */

    for (int k = 0; k < 3; k++) {
        double s = sqrt(lambda[k] > 0.0 ? lambda[k] : 0.0);
        for (int j = 0; j < 3; j++)
            ctrl[(k+1)*3 + j] = centroid[j] + s * V[k*3 + j];
    }

    /* 修复退化: 若 PCA 有重复特征值, 控制点 c3,c4 可能重合
     * 用叉积生成第三个独立方向 */
    {
        double *c2 = &ctrl[3], *c3 = &ctrl[6], *c4 = &ctrl[9];
        double d23 = 0, d24 = 0, d34 = 0;
        for (int d = 0; d < 3; d++) {
            double t = c2[d] - c3[d]; d23 += t*t;
            t = c2[d] - c4[d];        d24 += t*t;
            t = c3[d] - c4[d];        d34 += t*t;
        }
        double eps_d = 1e-10;
        if (d23 < eps_d || d24 < eps_d || d34 < eps_d) {
            /* 用 c2-c1, c3-c1 叉积构造第三个独立方向 */
            double v1[3], v2[3], v3[3];
            for (int d = 0; d < 3; d++) {
                v1[d] = c2[d] - centroid[d];
                v2[d] = c3[d] - centroid[d];
            }
            vec_cross(v1, v2, v3);
            double n3 = sqrt(vec_dot(v3, v3, 3));
            if (n3 > 1e-12) {
                double s = sqrt(lambda[2] > 0.0 ? lambda[2] : lambda[0] * 0.1);
                if (s < 1e-10) s = sqrt(lambda[0] > 0.0 ? lambda[0] : 1.0) * 0.1;
                for (int d = 0; d < 3; d++)
                    c4[d] = centroid[d] + s * v3[d] / n3;
            }
        }
    }
}

/* ========== 步骤 2: 计算重心坐标 α ========== */

static void compute_barycentric(const double *pts3D, int n,
                                 const double ctrl[12],
                                 double alpha[PNP_MAX_POINTS * 4])
{
    /* 构建 4×4 系数矩阵 A = [c1 c2 c3 c4; 1 1 1 1] */
    double A[16];
    for (int j = 0; j < 4; j++) {
        A[0  + j] = ctrl[j*3 + 0];  /* row 0: x_j */
        A[4  + j] = ctrl[j*3 + 1];  /* row 1: y_j */
        A[8  + j] = ctrl[j*3 + 2];  /* row 2: z_j */
        A[12 + j] = 1.0;            /* row 3: 1   */
    }

    for (int i = 0; i < n; i++) {
        double Acopy[16];
        mat_copy(A, Acopy, 4, 4);
        double b[4] = {pts3D[i*3], pts3D[i*3+1], pts3D[i*3+2], 1.0};
        mat_solve(Acopy, &alpha[i*4], b, 4);
    }
}

/* ========== 步骤 3: 构建 M 矩阵 (2n × 12) ========== */

static void build_M(const double *pts2D, const double *alpha, int n,
                     double *M)
{
    for (int i = 0; i < n; i++) {
        double u = pts2D[i*2], v = pts2D[i*2 + 1];
        const double *a = &alpha[i*4];
        int row0 = (2*i)   * 12;
        int row1 = (2*i+1) * 12;

        for (int j = 0; j < 4; j++) {
            int c0 = j * 3;
            M[row0 + c0 + 0] =  a[j];
            M[row0 + c0 + 1] =  0.0;
            M[row0 + c0 + 2] = -u * a[j];

            M[row1 + c0 + 0] =  0.0;
            M[row1 + c0 + 1] =  a[j];
            M[row1 + c0 + 2] = -v * a[j];
        }
    }
}

/* ========== 步骤 4: M^T*M 零空间 (mat_power_inv + 紧缩) ========== */

static int compute_null_space(const double *M, int rows,
                               double eigvecs[48], double eigvals[4])
{
    /* C = M^T * M (12×12) */
    double C[144] = {0};
    int cols = 12;
    for (int i = 0; i < rows; i++) {
        const double *row = &M[i * cols];
        for (int p = 0; p < cols; p++) {
            double mp = row[p];
            if (fabs(mp) < 1e-20) continue;
            int base = p * cols;
            for (int q = p; q < cols; q++)
                C[base + q] += mp * row[q];
        }
    }
    /* 对称化下三角 */
    for (int p = 0; p < cols; p++)
        for (int q = 0; q < p; q++)
            C[p*cols + q] = C[q*cols + p];

    /* 紧缩法求 N 个最小特征对 */
    int N = 0;
    for (int k = 0; k < 4; k++) {
        double *vk = &eigvecs[k * 12];
        /* 初始猜测 */
        for (int i = 0; i < 12; i++)
            vk[i] = (double)((i * 17 + k * 31 + 7) % 1000) / 1000.0;

        double work[144];
        double lam = mat_power_inv(C, vk, work, 12, 0.0, 80, 1e-12);
        eigvals[k] = lam;

        /* 归一化 vk */
        double nrm2 = vec_dot(vk, vk, 12);
        if (nrm2 < 1e-20) break;
        double inv_nrm = 1.0 / sqrt(nrm2);
        for (int i = 0; i < 12; i++) vk[i] *= inv_nrm;

        N = k + 1;

        /* 紧缩: C = C - λ * v * v^T */
        mat_rank1_update(C, vk, -lam, 12);

        /* λ 显著增大 → 停止 (至少取 1 个) */
        if (k > 0 && lam > eigvals[0] * 500.0) break;
    }
    return N;
}

/* ========== 步骤 5: β 优化 (lm_solve) ========== */

typedef struct {
    const double *eigvecs;    /* N 个 12 维向量               */
    const double *ctrl;       /* 4 个世界系控制点 (12 doubles)*/
    int N;                    /* 零空间维数                   */
    double dist2_w[6];        /* 世界系控制点距离平方 (预计算) */
} beta_data_t;

static void beta_residual(const double *beta, void *data,
                           double *r, int m, int n_)
{
    (void)n_;
    beta_data_t *bd = (beta_data_t *)data;
    int N = bd->N;

    /* x_cam = Σ β_k * v_k */
    double x_cam[12] = {0};
    for (int k = 0; k < N; k++)
        for (int p = 0; p < 12; p++)
            x_cam[p] += beta[k] * bd->eigvecs[k*12 + p];

    /* 控制点相机坐标 c_j^c = x_cam[j*3 : j*3+3] */
    double cb[4][3];
    for (int j = 0; j < 4; j++)
        for (int d = 0; d < 3; d++)
            cb[j][d] = x_cam[j*3 + d];

    /* 6 对距离残差 */
    int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int p = 0; p < m; p++) {
        int a = pairs[p][0], b = pairs[p][1];
        double dx = cb[a][0] - cb[b][0];
        double dy = cb[a][1] - cb[b][1];
        double dz = cb[a][2] - cb[b][2];
        double d2 = dx*dx + dy*dy + dz*dz;
        r[p] = d2 - bd->dist2_w[p];
    }
}

/* ========== 步骤 8: Horn 绝对定向 ========== */

static void absolute_orientation(const double *pts3D_w,
                                  const double *pts3D_c,
                                  int n, double R[9], double t[3])
{
    /* 质心 */
    double cw[3] = {0}, cc[3] = {0};
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            cw[d] += pts3D_w[i*3 + d];
            cc[d] += pts3D_c[i*3 + d];
        }
    }
    double inv_n = 1.0 / n;
    for (int d = 0; d < 3; d++) { cw[d] *= inv_n; cc[d] *= inv_n; }

    /* 互协方差 H = Σ (pc_i - cc) * (pw_i - cw)^T */
    double H[9] = {0};
    for (int i = 0; i < n; i++) {
        double dw[3], dc[3];
        for (int d = 0; d < 3; d++) {
            dw[d] = pts3D_w[i*3 + d] - cw[d];
            dc[d] = pts3D_c[i*3 + d] - cc[d];
        }
        H[0] += dc[0]*dw[0]; H[1] += dc[0]*dw[1]; H[2] += dc[0]*dw[2];
        H[3] += dc[1]*dw[0]; H[4] += dc[1]*dw[1]; H[5] += dc[1]*dw[2];
        H[6] += dc[2]*dw[0]; H[7] += dc[2]*dw[1]; H[8] += dc[2]*dw[2];
    }

    /* SVD */
    double U[9], S[3], V[9];
    mat33_svd(H, U, S, V);

    /* R = V * U^T */
    double Ut[9];
    mat_transpose(U, Ut, 3, 3);
    mat_mul(V, Ut, R, 3, 3, 3);

    /* 确保 det(R) = +1 (非反射) */
    double detR = R[0]*(R[4]*R[8]-R[5]*R[7])
                - R[1]*(R[3]*R[8]-R[5]*R[6])
                + R[2]*(R[3]*R[7]-R[4]*R[6]);
    if (detR < 0.0) {
        /* V 最后一列取反 → R = V * diag(1,1,-1) * U^T */
        V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8];
        mat_mul(V, Ut, R, 3, 3, 3);
    }

    /* t = cc - R * cw */
    double Rcw[3];
    mat_vec_mul(R, cw, Rcw, 3, 3);
    for (int d = 0; d < 3; d++) t[d] = cc[d] - Rcw[d];
}

/* ========== 步骤 9: 重投影误差 (LM 精化) ========== */

typedef struct {
    const double *pts3D;
    const double *pts2D;
    int n;
    double fx, fy, cx, cy;
} reproj_data_t;

/* 残差: 2*n 维 (每点 u, v 误差) */
static void reproj_residual(const double *pose, void *data,
                             double *r, int m, int n)
{
    (void)m; (void)n;
    reproj_data_t *d = (reproj_data_t *)data;

    /* pose = [rx, ry, rz, tx, ty, tz] */
    double R[9];
    mat33_exp_so3(pose, R);

    for (int i = 0; i < d->n; i++) {
        /* P_cam = R * P_w + t */
        double pc[3];
        mat_vec_mul(R, &d->pts3D[i*3], pc, 3, 3);
        pc[0] += pose[3]; pc[1] += pose[4]; pc[2] += pose[5];

        double inv_z = 1.0 / (fabs(pc[2]) > 1e-10 ? pc[2] : 1e-10);
        double u_proj = d->fx * pc[0] * inv_z + d->cx;
        double v_proj = d->fy * pc[1] * inv_z + d->cy;

        r[2*i]   = u_proj - d->pts2D[2*i];
        r[2*i+1] = v_proj - d->pts2D[2*i+1];
    }
}

/* ========== 主求解器 ========== */

int pnp_solve(const double *points_3D, const double *points_2D,
              int n, const pnp_camera_t *cam,
              pnp_result_t *result)
{
    if (n < 4 || n > PNP_MAX_POINTS || !cam || !result) return -1;

    double fx = cam->fx, fy = cam->fy;
    double cx = cam->cx, cy = cam->cy;

    /* --- 转换像素坐标 → 归一化图像坐标 --- */
    double pts2D_norm[PNP_MAX_POINTS * 2];
    for (int i = 0; i < n; i++) {
        pts2D_norm[2*i]   = (points_2D[2*i]   - cx) / fx;
        pts2D_norm[2*i+1] = (points_2D[2*i+1] - cy) / fy;
    }

    /* ---- 步骤 1: 选择控制点 ---- */
    double ctrl_w[12];
    select_control_points(points_3D, n, ctrl_w);

    /* ---- 步骤 2: 重心坐标 ---- */
    double alpha[PNP_MAX_POINTS * 4];
    compute_barycentric(points_3D, n, ctrl_w, alpha);

    /* ---- 步骤 3: M 矩阵 ---- */
    double M[PNP_MAX_POINTS * 2 * 12];  /* 最多 40×12 */
    build_M(pts2D_norm, alpha, n, M);
    int rows = 2 * n;

    /* ---- 步骤 4: 零空间 ---- */
    double eigvecs[48];    /* 最多 4 × 12 */
    double eigvals[4];
    int N = compute_null_space(M, rows, eigvecs, eigvals);

    /* ---- 步骤 5: β 优化 ---- */
    beta_data_t bd;
    bd.eigvecs = eigvecs;
    bd.ctrl    = ctrl_w;
    bd.N       = N;

    /* 预计算世界系控制点距离 */
    int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int p = 0; p < 6; p++) {
        int a = pairs[p][0], b = pairs[p][1];
        double dx = ctrl_w[a*3+0] - ctrl_w[b*3+0];
        double dy = ctrl_w[a*3+1] - ctrl_w[b*3+1];
        double dz = ctrl_w[a*3+2] - ctrl_w[b*3+2];
        bd.dist2_w[p] = dx*dx + dy*dy + dz*dz;
    }

    /* β 初值: [1, 0, ..., 0] */
    double beta[4] = {1.0, 0.0, 0.0, 0.0};

    if (N == 1) {
        /* N=1: β₀ 由距离约束直接算 */
        double sum_dw = 0.0, sum_dv = 0.0;
        for (int p = 0; p < 6; p++) {
            /* 用第一个零空间向量重建控制点 */
            double cb[4][3] = {{0}};
            for (int j = 0; j < 4; j++)
                for (int d = 0; d < 3; d++)
                    cb[j][d] = eigvecs[j*3 + d];
            int a = pairs[p][0], b = pairs[p][1];
            double dx = cb[a][0]-cb[b][0];
            double dy = cb[a][1]-cb[b][1];
            double dz = cb[a][2]-cb[b][2];
            double dv2 = dx*dx + dy*dy + dz*dz;
            if (dv2 > 1e-20) {
                sum_dw += bd.dist2_w[p];
                sum_dv += dv2;
            }
        }
        beta[0] = sqrt(sum_dw / (sum_dv > 1e-20 ? sum_dv : 1e-20));
    } else {
        lm_config_t cfg;
        cfg.max_iter = 50; cfg.tau = 1e-3;
        cfg.eps_grad = 1e-10; cfg.eps_step = 1e-10; cfg.eps_cost = 1e-10;
        cfg.lambda_init = -1.0;
        lm_solve(beta, 6, N, beta_residual, NULL, &bd, &cfg, NULL);
    }

    /* ---- 步骤 6: 恢复控制点相机坐标 ---- */
    double ctrl_c[12] = {0};
    for (int k = 0; k < N; k++)
        for (int j = 0; j < 12; j++)
            ctrl_c[j] += beta[k] * eigvecs[k*12 + j];

    /* ---- 步骤 7: 恢复所有 3D 点的相机坐标 ---- */
    double pts3D_c[PNP_MAX_POINTS * 3];
    for (int i = 0; i < n; i++) {
        pts3D_c[i*3]   = 0.0;
        pts3D_c[i*3+1] = 0.0;
        pts3D_c[i*3+2] = 0.0;
        for (int j = 0; j < 4; j++) {
            double a = alpha[i*4 + j];
            pts3D_c[i*3]   += a * ctrl_c[j*3];
            pts3D_c[i*3+1] += a * ctrl_c[j*3+1];
            pts3D_c[i*3+2] += a * ctrl_c[j*3+2];
        }
    }

    /* 修正符号: 确保大部分点的 Z > 0 (在相机前方) */
    {
        int neg_z = 0;
        for (int i = 0; i < n; i++)
            if (pts3D_c[i*3 + 2] < 0.0) neg_z++;
        if (neg_z > n / 2) {
            for (int i = 0; i < n * 3; i++)
                pts3D_c[i] = -pts3D_c[i];
        }
    }

    /* ---- 步骤 8: 绝对定向 ---- */
    double R[9], t[3];
    absolute_orientation(points_3D, pts3D_c, n, R, t);

    /* ---- 步骤 9: LM 重投影精化 ---- */
    double w[3];
    mat33_log_so3(R, w);
    double pose[6] = {w[0], w[1], w[2], t[0], t[1], t[2]};

    reproj_data_t rdata;
    rdata.pts3D = points_3D;
    rdata.pts2D = points_2D;
    rdata.n     = n;
    rdata.fx = fx; rdata.fy = fy;
    rdata.cx = cx; rdata.cy = cy;

    lm_config_t cfg;
    cfg.max_iter = 30; cfg.tau = 1e-3;
    cfg.eps_grad = 1e-10; cfg.eps_step = 1e-10; cfg.eps_cost = 1e-12;
    cfg.lambda_init = -1.0;

    lm_result_t lmr;
    int lr = lm_solve(pose, 2*n, 6, reproj_residual, NULL, &rdata, &cfg, &lmr);

    /* 最终 R,t */
    mat33_exp_so3(pose, R);
    t[0] = pose[3]; t[1] = pose[4]; t[2] = pose[5];

    memcpy(result->R, R, 9 * sizeof(double));
    memcpy(result->t, t, 3 * sizeof(double));
    result->iterations  = lmr.iterations;
    result->reproj_err  = sqrt(2.0 * lmr.final_cost / n);  /* RMS */

    return (lr == 0 || lr == -1) ? 0 : -1;
}
