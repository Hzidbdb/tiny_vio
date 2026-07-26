
/*
 * @Author: garygoo
 * @Date: 2026-06-09 05:29:45
 * @LastEditors: garygoo 
 * @LastEditTime: 2026-07-27 05:59:30
 * @Description: 线代的微缩版本，用于之后的guass-newton法的实现以及
 EKF融合与优化问题的求解
 */
#include "stdio.h"
#include "string.h"
#include "tiny_linalg.h"
#include "math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//restrict加多路向量点积计算
double vec_dot(const double *restrict a,const double *restrict b,int n){
    double s0 =0.0, s1 =0.0, s2 =0.0, s3 =0.0;
    int i=0;
    for(;i+3<n;i+=4){
        s0 += a[i] * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    for(;i<n;i++){
        s0 += a[i] * b[i];
    }
  return s0 + s1 + s2 + s3;
}
//向量加法
void vec_add(const double *restrict a, const double *restrict b, double *restrict out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
//向量减法
void vec_remove(const double *restrict a,const double *restrict b,double *restrict out,int n){
  for(int i=0;i<n;i++){
    out[i]=a[i]-b[i];
  }
}
//向量叉积
 void vec_cross(const double a[3], const double b[3], double out[3])
  {
      out[0] = a[1] * b[2] - a[2] * b[1];
      out[1] = a[2] * b[0] - a[0] * b[2];
      out[2] = a[0] * b[1] - a[1] * b[0];
  }

  //L2范数
  double vec_L2(const double *restrict a, int n) {
     double sum = sqrt(vec_dot(a, a, n));
     return sum;
  }

// y = A * x   A[m×n], x[n], y[m] 矩阵向量乘法
void mat_vec_mul(const double *restrict A, const double *restrict x,
                 double *restrict y, int m, int n)
{
    for (int i = 0; i < m; i++) {
        y[i] = vec_dot(A + i * n, x, n);
    }
}

// C = A * B   A[m×k], B[k×n], C[m×n]矩阵乘法
void mat_mul(const double *restrict A, const double *restrict B,
             double *restrict C, int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) {
                s += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = s;
        }
    }
}

// B = A^T   A[m×n], B[n×m] 转置矩阵
void mat_transpose(const double *restrict A, double *restrict B, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

// C = A + B A[m×n], B[m×n], C[m×n]矩阵加法
void mat_add(const double *restrict A, const double *restrict B,
             double *restrict C, int m, int n)
{
    int total = m * n;
    for (int i = 0; i < total; i++) {
        C[i] = A[i] + B[i];
    }
}

// C = A - B A[m×n], B[m×n], C[m×n]矩阵减法
void mat_sub(const double *restrict A, const double *restrict B,
             double *restrict C, int m, int n)
{
    int total = m * n;
    for (int i = 0; i < total; i++) {
        C[i] = A[i] - B[i];
    }
}

// out = A * x + y    A[m×n], x[n], y[m], out[m] 矩阵乘加
void mat_vec_madd(const double *restrict A, const double *restrict x,
                  const double *restrict y, double *restrict out, int m, int n)
{
    for (int i = 0; i < m; i++) {
        out[i] = vec_dot(A + i * n, x, n) + y[i];
    }
}

// A = A + alpha * x * x^T   rank-1 update A[n×n], x[n], alpha标量
void mat_rank1_update(double *restrict A, const double *restrict x,
                      double alpha, int n)
{
    for (int i = 0; i < n; i++) {
        double ai = alpha * x[i];
        for (int j = 0; j < n; j++) {
            A[i * n + j] += ai * x[j];
        }
    }
}


// 高斯消元法求解线性方程组 Ax = b，A是n×n矩阵，x是未知向量，b是常数向量
static int find_pivot(const double *A, int n, int col)
{
    int best = col;
    double best_val = fabs(A[col * n + col]);
    for (int row = col + 1; row < n; row++) {
        double v = fabs(A[row * n + col]);
        if (v > best_val) {
            best_val = v;
            best = row;
        }
    }
    return best;
}
// 交换矩阵A的两行以及对应的向量b的元素
static void swap_rows(double *A, double *b, int n, int r1, int r2)
{
    if (r1 == r2) return;
    for (int j = 0; j < n; j++) {
        double tmp = A[r1 * n + j];
        A[r1 * n + j] = A[r2 * n + j];
        A[r2 * n + j] = tmp;
    }
    double tmp = b[r1];
    b[r1] = b[r2];
    b[r2] = tmp;
}
// 返回0表示成功，-1表示矩阵奇异
int mat_solve(double *A, double *x, const double *b, int n)
{
    // copy b into x (will be transformed into the solution)
    memcpy(x, b, (size_t)n * sizeof(double));

    // 高斯消元法
    for (int col = 0; col < n; col++) 
    {
        int pivot = find_pivot(A, n, col);
        if (fabs(A[pivot * n + col]) < 1e-15) return -1;  // singular
        swap_rows(A, x, n, col, pivot);

        double piv = A[col * n + col];
        for (int row = col + 1; row < n; row++) 
        {
            double factor = A[row * n + col] / piv;
            A[row * n + col] = 0.0;
            for (int j = col + 1; j < n; j++) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            x[row] -= factor * x[col];
        }
    }

    // 返回解
    for (int i = n - 1; i >= 0; i--) 
    {
        double s = x[i];
        for (int j = i + 1; j < n; j++) 
        {
            s -= A[i * n + j] * x[j];
        }
        x[i] = s / A[i * n + i];
    }
    return 0;

}

// C = A (拷贝 m×n 矩阵)
void mat_copy(const double *A, double *C, int m, int n)
{
    memcpy(C, A, (size_t)(m * n) * sizeof(double));
}

// A = I_n (生成 n×n 单位矩阵)
void mat_identity(double *A, int n)
{
    memset(A, 0, (size_t)(n * n) * sizeof(double));
    for (int i = 0; i < n; i++)
        A[i * n + i] = 1.0;
}

// 取对角元
void mat_get_diag(const double *A, double *diag, int n)
{
    for (int i = 0; i < n; i++)
        diag[i] = A[i * n + i];
}

// 设对角元
void mat_set_diag(double *A, const double *diag, int n)
{
    for (int i = 0; i < n; i++)
        A[i * n + i] = diag[i];
}

int mat_cholesky(const double *A, double *L,int n)
{
    //初始化L为0矩阵
    for(int i=0;i<n*n;i++)
    {
        L[i]=0.0;
    }
    for(int j=0;j<n;j++)
    {
        //使用vec_dot计算对角线元素
        double sum = vec_dot(L+j*n, L+j*n, j);
        double diag = A[j*n+j] - sum;
        if(diag<=1e-15)
        {
            return -1; // not positive definite
        }
        L[j*n+j] = sqrt(diag);


         //计算下三角矩阵的非对角线元素
        for(int i=j+1;i<n;i++)
        {
            //使用vec_dot计算非对角线元素
            sum = vec_dot(L+i*n, L+j*n, j);
            L[i*n+j] = (A[i*n+j] - sum) / L[j*n+j];
        }
    }
    return 0; // success

}

/* ========== 3×3 对称矩阵特征分解 (三次方程解析解) ========== */

// 解三次方程 x^3 + a*x^2 + b*x + c = 0, 三个实根 (对称矩阵保证)
static void solve_cubic(double a, double b, double c, double roots[3])
{
    double p = b - a*a/3.0;
    double q = c - a*b/3.0 + 2.0*a*a*a/27.0;
    double disc = p*p*p/(-27.0);  // = -p³/27, ≥0 对应三实根

    if (disc <= 0) disc = 0.0;
    double acos_arg = -q * 0.5 / sqrt(disc > 1e-30 ? disc : 1e-30);
    if (acos_arg >  1.0) acos_arg =  1.0;
    if (acos_arg < -1.0) acos_arg = -1.0;
    double phi = acos(acos_arg);
    double r_arg = -p / 3.0;
    if (r_arg < 0.0) r_arg = 0.0;
    double r = 2.0 * sqrt(r_arg);
    double shift = a / 3.0;

    for (int k = 0; k < 3; k++)
        roots[k] = r * cos((phi + 2.0 * M_PI * k) / 3.0) - shift;

    // 降序排列
    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (roots[i] < roots[j]) {
                double t = roots[i]; roots[i] = roots[j]; roots[j] = t;
            }
}

// 从 (A - lambda*I) 中提取特征向量 (行叉积法)
static void eigenvec_from_lambda(const double A[9], double lambda, double v[3])
{
    double B[9];
    mat_copy(A, B, 3, 3);
    B[0] -= lambda; B[4] -= lambda; B[8] -= lambda;

    // 取范数最大的两行叉积作为零空间方向
    double r0[3] = {B[0], B[1], B[2]};
    double r1[3] = {B[3], B[4], B[5]};
    double r2[3] = {B[6], B[7], B[8]};
    double n0 = vec_dot(r0, r0, 3);
    double n1 = vec_dot(r1, r1, 3);
    double n2 = vec_dot(r2, r2, 3);

    if (n0 >= n1 && n0 >= n2) {
        if (n1 >= n2) vec_cross(r0, r1, v);
        else          vec_cross(r0, r2, v);
    } else if (n1 >= n0 && n1 >= n2) {
        if (n0 >= n2) vec_cross(r1, r0, v);
        else          vec_cross(r1, r2, v);
    } else {
        if (n0 >= n1) vec_cross(r2, r0, v);
        else          vec_cross(r2, r1, v);
    }

    // 归一化
    double len = sqrt(vec_dot(v, v, 3));
    if (len < 1e-15) {  // 退化: 随便取单位向量
        v[0] = 1.0; v[1] = 0.0; v[2] = 0.0;
    } else {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
}

void mat33_symeig(const double A[9], double lambda[3], double V[9])
{
    // 特征多项式: det(A - λI) = -λ³ + p·λ² - q·λ + r = 0
    // 等价于 λ³ - p·λ² + q·λ - r = 0
    double a11 = A[0], a22 = A[4], a33 = A[8];
    double a12 = A[1], a13 = A[2], a23 = A[5];

    double p = a11 + a22 + a33;
    double q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23;
    double r = a11*a22*a33 + 2.0*a12*a13*a23
             - a11*a23*a23 - a22*a13*a13 - a33*a12*a12;

    solve_cubic(-p, q, -r, lambda);

    // 对每个特征值求特征向量
    for (int k = 0; k < 3; k++)
        eigenvec_from_lambda(A, lambda[k], &V[k*3]);
}

/* ========== 逆幂迭代 (n×n 对称矩阵最小特征值) ========== */

double mat_power_inv(const double *A, double *v, double *work,
                     int n, double sigma, int max_iter, double tol)
{
    double lambda_prev = sigma;

    for (int iter = 0; iter < max_iter; iter++) {
        // 复制 A - sigma*I 到 work (mat_solve 会破坏)
        mat_copy(A, work, n, n);
        for (int i = 0; i < n; i++)
            work[i * n + i] -= sigma;

        double b[24];  // 导航场景 n ≤ 12, 栈上限 24 够用
        double x[24];
        memcpy(b, v, (size_t)n * sizeof(double));

        int ret = mat_solve(work, x, b, n);
        if (ret != 0) {
            // 加微小扰动恢复可逆性
            mat_copy(A, work, n, n);
            for (int i = 0; i < n; i++)
                work[i * n + i] += 1e-12 - sigma;
            mat_solve(work, x, b, n);
        }

        // 归一化 x → v
        double nx = sqrt(vec_dot(x, x, n));
        if (nx < 1e-15) return sigma;
        for (int i = 0; i < n; i++) v[i] = x[i] / nx;

        // Rayleigh 商: 对对称 A, λ = v^T*A*v / v^T*v  (v 已归一化)
        double Av[24];
        mat_vec_mul(A, v, Av, n, n);
        double lambda = vec_dot(v, Av, n);

        if (fabs(lambda - lambda_prev) < tol)
            return lambda;
        lambda_prev = lambda;
    }
    return lambda_prev;
}

/* ========== 3×3 SVD 闭式解 ========== */

int mat33_svd(const double A[9], double U[9], double S[3], double V[9])
{
    // 1. A^T * A (3×3 对称半正定)
    double AtA[9] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                AtA[i*3 + j] += A[k*3 + i] * A[k*3 + j];

    // 2. 特征分解 AtA → V, lambda
    double lambda[3];
    mat33_symeig(AtA, lambda, V);

    // 3. 奇异值 = sqrt(eigenvalue)
    for (int i = 0; i < 3; i++) {
        if (lambda[i] > 1e-15)
            S[i] = sqrt(lambda[i]);
        else
            S[i] = 0.0;
    }

    // 4. U = A * V * diag(1/S)
    double AV[9];
    mat_mul(A, V, AV, 3, 3, 3);
    for (int j = 0; j < 3; j++) {
        double inv_s = (S[j] > 1e-15) ? 1.0 / S[j] : 0.0;
        for (int i = 0; i < 3; i++)
            U[i*3 + j] = AV[i*3 + j] * inv_s;
    }

    // 补全 U 的列 (处理 S[j]=0 的情况)
    for (int j = 0; j < 3; j++) {
        if (S[j] < 1e-15) {
            if (j == 0) {
                U[0] = 1.0; U[3] = 0.0; U[6] = 0.0;
            } else if (j == 1) {
                double ref[3];
                if (fabs(U[0]) < 0.9) { ref[0]=1.0; ref[1]=0.0; ref[2]=0.0; }
                else                   { ref[0]=0.0; ref[1]=1.0; ref[2]=0.0; }
                vec_cross(&U[0], ref, &U[j*3]);
                double n = sqrt(vec_dot(&U[j*3], &U[j*3], 3));
                U[j*3] /= n; U[j*3+1] /= n; U[j*3+2] /= n;
            } else {
                vec_cross(&U[0], &U[3], &U[j*3]);
                double n = sqrt(vec_dot(&U[j*3], &U[j*3], 3));
                if (n < 1e-10) {
                    U[j*3] = 1.0; U[j*3+1] = 0.0; U[j*3+2] = 0.0;
                } else {
                    U[j*3] /= n; U[j*3+1] /= n; U[j*3+2] /= n;
                }
            }
        }
    }
    return 0;
}

// 前代+回代: L*L^T * x = b → x
void mat_cholesky_solve(const double *L, const double *b, double *x, int n)
{
    double y[64];  /* max n = LM_MAX_PARAMS (48) */

    // 前代 L*y = b
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++)
            s -= L[i * n + j] * y[j];
        y[i] = s / L[i * n + i];
    }
    // 回代 L^T*x = y
    for (int i = n - 1; i >= 0; i--) {
        double s = y[i];
        for (int j = i + 1; j < n; j++)
            s -= L[j * n + i] * x[j];
        x[i] = s / L[i * n + i];
    }
}
//================== SO(3) 指数映射与对数映射 ==================
void mat33_exp_so3(const double w[3], double R[9])
{
    double theta2 = vec_dot(w, w, 3);
    if (theta2 < 1e-20) {
        R[0]=1.0;  R[1]=-w[2]; R[2]=w[1];
        R[3]=w[2]; R[4]=1.0;   R[5]=-w[0];
        R[6]=-w[1];R[7]=w[0];  R[8]=1.0;
        return;
    }
    double theta = sqrt(theta2);
    double k0=w[0]/theta, k1=w[1]/theta, k2=w[2]/theta;
    double s=sin(theta), c=cos(theta), t=1.0-c;
    R[0]=c+t*k0*k0;     R[1]=t*k0*k1-s*k2; R[2]=t*k0*k2+s*k1;
    R[3]=t*k0*k1+s*k2;  R[4]=c+t*k1*k1;    R[5]=t*k1*k2-s*k0;
    R[6]=t*k0*k2-s*k1;  R[7]=t*k1*k2+s*k0; R[8]=c+t*k2*k2;
}
//================== SO(3) 对数映射 ==================
void mat33_log_so3(const double R[9], double w[3])
{
    double cos_t = (R[0]+R[4]+R[8]-1.0)*0.5;
    if (cos_t >  1.0) cos_t =  1.0;
    if (cos_t < -1.0) cos_t = -1.0;
    double theta = acos(cos_t);

    if (theta < 1e-10) {
        w[0]=(R[7]-R[5])*0.5; w[1]=(R[2]-R[6])*0.5; w[2]=(R[3]-R[1])*0.5;
        return;
    }
    if (fabs(theta - M_PI) < 1e-6) {
        double d0=(R[0]+1.0)*0.5, d1=(R[4]+1.0)*0.5, d2=(R[8]+1.0)*0.5;
        double k0, k1, k2;
        if (d0>=d1 && d0>=d2) {
            k0=sqrt(d0>0?d0:0.0);
            k1=(R[1]+R[3])*0.25/k0; k2=(R[2]+R[6])*0.25/k0;
        } else if (d1>=d2) {
            k1=sqrt(d1>0?d1:0.0);
            k0=(R[1]+R[3])*0.25/k1; k2=(R[5]+R[7])*0.25/k1;
        } else {
            k2=sqrt(d2>0?d2:0.0);
            k0=(R[2]+R[6])*0.25/k2; k1=(R[5]+R[7])*0.25/k2;
        }
        w[0]=k0*theta; w[1]=k1*theta; w[2]=k2*theta;
        return;
    }
    double f = theta / (2.0*sin(theta));
    w[0]=(R[7]-R[5])*f; w[1]=(R[2]-R[6])*f; w[2]=(R[3]-R[1])*f;
}

/* SO(3) 右 Jacobian: Jr(θ) = (sinθ/θ)·I + (1-sinθ/θ)·aa^T - ((1-cosθ)/θ)·skew(a)
 * 其中 θ = |w|, a = w/θ */
void mat33_right_jacobian(const double w[3], double Jr[9])
{
    double t2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    double t  = sqrt(t2);
    if (t < 1e-12) {
        Jr[0]=1.0; Jr[1]=0.0; Jr[2]=0.0;
        Jr[3]=0.0; Jr[4]=1.0; Jr[5]=0.0;
        Jr[6]=0.0; Jr[7]=0.0; Jr[8]=1.0;
        return;
    }
    double a[3] = {w[0]/t, w[1]/t, w[2]/t};
    double st = sin(t), ct = cos(t);
    double A = st / t;
    double B = 1.0 - A;
    double C = (1.0 - ct) / t;

    Jr[0] = A + B*a[0]*a[0];          Jr[1] =     B*a[0]*a[1] + C*a[2];  Jr[2] =     B*a[0]*a[2] - C*a[1];
    Jr[3] =     B*a[1]*a[0] - C*a[2]; Jr[4] = A + B*a[1]*a[1];          Jr[5] =     B*a[1]*a[2] + C*a[0];
    Jr[6] =     B*a[2]*a[0] + C*a[1]; Jr[7] =     B*a[2]*a[1] - C*a[0]; Jr[8] = A + B*a[2]*a[2];
}

/* SO(3) 右 Jacobian 逆: Jr^{-1}(θ) = (θ/2·cot(θ/2))·I + (1-θ/2·cot(θ/2))·aa^T - (θ/2)·skew(a) */
void mat33_right_jacobian_inv(const double w[3], double Jr_inv[9])
{
    double t2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    double t  = sqrt(t2);
    if (t < 1e-12) {
        Jr_inv[0]=1.0; Jr_inv[1]=0.0; Jr_inv[2]=0.0;
        Jr_inv[3]=0.0; Jr_inv[4]=1.0; Jr_inv[5]=0.0;
        Jr_inv[6]=0.0; Jr_inv[7]=0.0; Jr_inv[8]=1.0;
        return;
    }
    double a[3] = {w[0]/t, w[1]/t, w[2]/t};
    double half_t = 0.5 * t;
    double A = half_t / tan(half_t);
    double B = 1.0 - A;
    double C = half_t;

    Jr_inv[0] = A + B*a[0]*a[0];          Jr_inv[1] =     B*a[0]*a[1] + C*a[2];  Jr_inv[2] =     B*a[0]*a[2] - C*a[1];
    Jr_inv[3] =     B*a[1]*a[0] - C*a[2]; Jr_inv[4] = A + B*a[1]*a[1];          Jr_inv[5] =     B*a[1]*a[2] + C*a[0];
    Jr_inv[6] =     B*a[2]*a[0] + C*a[1]; Jr_inv[7] =     B*a[2]*a[1] - C*a[0]; Jr_inv[8] = A + B*a[2]*a[2];
}