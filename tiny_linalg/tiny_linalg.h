#ifndef TINY_LINALG_H
#define TINY_LINALG_H

//向量运算

    //多路向量点积计算
    double vec_dot(const double *restrict a,const double *restrict b,int n);
    //向量加法
    void vec_add(const double *restrict a, const double *restrict b, double *restrict out, int n);
    //向量减法
    void vec_remove(const double *restrict a,const double *restrict b,double *restrict out,int n);
    //向量叉积
    void vec_cross(const double a[3], const double b[3], double out[3]);
    //L2范数
    double vec_L2(const double *restrict a, int n);

//矩阵运算

    // y = A * x  (A: m rows, n cols)
    void mat_vec_mul(const double *restrict A, const double *restrict x,
                    double *restrict y, int m, int n);

    // C = A * B  (A: m×k, B: k×n, C: m×n)
    void mat_mul(const double *restrict A, const double *restrict B,
                double *restrict C, int m, int k, int n);

    // B = A^T  (A: m×n, B: n×m)
    void mat_transpose(const double *restrict A, double *restrict B, int m, int n);

    // C = A + B
    void mat_add(const double *restrict A, const double *restrict B,
                double *restrict C, int m, int n);

    // C = A - B
    void mat_sub(const double *restrict A, const double *restrict B,
                double *restrict C, int m, int n);

    // out = A * x + y   (fused multiply-add, common in optimization)
    void mat_vec_madd(const double *restrict A, const double *restrict x,
                    const double *restrict y, double *restrict out, int m, int n);

    // solve Ax = b by Gaussian elimination with partial pivoting (A is n×n, in-place modified)
    // returns 0 on success, -1 if singular
    int mat_solve(double *A, double *x, const double *b, int n);

    // A = A + alpha * x * x^T   (rank-1 update, common in quasi-Newton / normal equations)
    void mat_rank1_update(double *restrict A, const double *restrict x,
                        double alpha, int n);

//Cholesky分解
    // Cholesky 分解 A = L * L^T (A 只读, L 输出下三角)
    // 返回 0 成功, -1 非正定
    int mat_cholesky(const double *A, double *L, int n);
    // 用 Cholesky 分解结果解 L*L^T * x = b → x
    void mat_cholesky_solve(const double *L, const double *b, double *x, int n);

// 特征分解 & SVD
    // 3×3 对称矩阵特征分解: A * V = V * diag(lambda), lambda 降序
    void mat33_symeig(const double A[9], double lambda[3], double V[9]);
    // 逆幂迭代: 求 A[n×n] 最接近 sigma 的特征值+特征向量
    // work: n×n 工作区, v: 初始猜测+输出特征向量 (归一化)
    double mat_power_inv(const double *A, double *v, double *work,
                         int n, double sigma, int max_iter, double tol);
    // 3×3 SVD: A = U * diag(S) * V^T, 闭式解
    int mat33_svd(const double A[9], double U[9], double S[3], double V[9]);
    // SO(3) 指数映射: 旋转向量 w → 旋转矩阵 R (罗德里格斯公式)
    void mat33_exp_so3(const double w[3], double R[9]);
    // SO(3) 对数映射: 旋转矩阵 R → 旋转向量 w
    void mat33_log_so3(const double R[9], double w[3]);
    // SO(3) 右 Jacobian: Jr(θ) = (sinθ/θ)·I + (1-sinθ/θ)·aa^T - ((1-cosθ)/θ)·skew(a)
    void mat33_right_jacobian(const double w[3], double Jr[9]);
    // SO(3) 右 Jacobian 逆: Jr⁻¹(θ) = (θ/2·cot(θ/2))·I + (1-θ/2·cot(θ/2))·aa^T - (θ/2)·skew(a)
    void mat33_right_jacobian_inv(const double w[3], double Jr_inv[9]);

// 工具函数
    // C = A   (拷贝 m×n 矩阵)
    void mat_copy(const double *A, double *C, int m, int n);
    // A = I_n (生成 n×n 单位矩阵)
    void mat_identity(double *A, int n);
    // 取/设对角元
    void mat_get_diag(const double *A, double *diag, int n);
    void mat_set_diag(double *A, const double *diag, int n);

#endif // TINY_LINALG_H