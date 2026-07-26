#include "stdio.h"
#include "math.h"
#include "tiny_opt.h"
#include "../tiny_linalg/tiny_linalg.h"

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL [%d]: %s\n", __LINE__, msg); } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(fabs((a)-(b)) < (tol), msg)

/* ================================================================
 * 测试 1: 线性最小二乘 y = a*x + b
 *
 * 已知最优解: a=2, b=3
 * 合成数据: y_i = 2*t_i + 3 + noise(0)
 * 参数: params[0]=a, params[1]=b
 * 残差: r_i = a*t_i + b - y_i
 * ================================================================ */

typedef struct { double t[10]; double y[10]; } linear_data_t;

static void linear_residual(const double *p, void *data,
                             double *r, int m, int n)
{
    (void)n;
    linear_data_t *d = (linear_data_t *)data;
    double a = p[0], b = p[1];
    for (int i = 0; i < m; i++)
        r[i] = a * d->t[i] + b - d->y[i];
}

static void linear_jacobian(const double *p, void *data,
                             double *J, int m, int n)
{
    (void)p; (void)n;
    linear_data_t *d = (linear_data_t *)data;
    /* J[i][0] = t_i, J[i][1] = 1.0 */
    for (int i = 0; i < m; i++) {
        J[i * 2 + 0] = d->t[i];
        J[i * 2 + 1] = 1.0;
    }
}

static void test_linear_ls(void)
{
    printf("linear least squares (analytic J) ...\n");

    linear_data_t d;
    int m = 10, n = 2;
    for (int i = 0; i < m; i++) {
        d.t[i] = (double)i;
        d.y[i] = 2.0 * d.t[i] + 3.0;   /* y = 2x+3 */
    }

    double x[] = {0.0, 0.0};  /* 初始猜测 */
    lm_result_t res;
    int ret = lm_solve(x, m, n, linear_residual, linear_jacobian,
                       &d, NULL, &res);

    CHECK(ret == 0, "linear converged");
    CHECK(res.iterations <= 10, "linear few iterations");
    CHECK_NEAR(x[0], 2.0, 1e-6, "linear a=2");
    CHECK_NEAR(x[1], 3.0, 1e-6, "linear b=3");
    CHECK_NEAR(res.final_cost, 0.0, 1e-12, "linear cost≈0");
}

/* ================================================================
 * 测试 2: Rosenbrock (解析 Jacobian)
 *
 * 残差: r0 = 1 - x, r1 = 10*(y - x^2)
 * 最优解: x=1, y=1, cost=0
 * 初始: (-1.2, 1.0)
 * ================================================================ */

static void rosenbrock_residual(const double *p, void *data,
                                 double *r, int m, int n)
{
    (void)data; (void)m; (void)n;
    double x = p[0], y = p[1];
    r[0] = 1.0 - x;
    r[1] = 10.0 * (y - x * x);
}

static void rosenbrock_jacobian(const double *p, void *data,
                                 double *J, int m, int n)
{
    (void)data; (void)m; (void)n;
    double x = p[0];
    /* J = [ -1       ,   0 ]
     *     [ -20*x    ,  10 ]  */
    J[0] = -1.0;          J[1] =  0.0;
    J[2] = -20.0 * x;     J[3] = 10.0;
}

static void test_rosenbrock_analytic(void)
{
    printf("Rosenbrock (analytic J) ...\n");

    double x[] = {-1.2, 1.0};
    lm_result_t res;
    int ret = lm_solve(x, 2, 2, rosenbrock_residual, rosenbrock_jacobian,
                       NULL, NULL, &res);

    CHECK(ret == 0, "rosenbrock converged");
    CHECK_NEAR(x[0], 1.0, 1e-6, "rosenbrock x=1");
    CHECK_NEAR(x[1], 1.0, 1e-5, "rosenbrock y=1");
    CHECK_NEAR(res.final_cost, 0.0, 1e-12, "rosenbrock cost≈0");
    printf("  iterations=%d, lambda_final=%.2e\n",
           res.iterations, res.lambda_final);
}

/* ================================================================
 * 测试 3: Rosenbrock (数值 Jacobian)
 *
 * 与测试 2 相同, 但不提供解析 Jacobian, 验证数值微分也能收敛。
 * ================================================================ */

static void test_rosenbrock_numerical(void)
{
    printf("Rosenbrock (numerical J) ...\n");

    double x[] = {-1.2, 1.0};
    lm_result_t res;
    int ret = lm_solve(x, 2, 2, rosenbrock_residual, NULL,
                       NULL, NULL, &res);

    CHECK(ret == 0, "rosenbrock_num converged");
    CHECK_NEAR(x[0], 1.0, 1e-5, "rosenbrock_num x=1");
    CHECK_NEAR(x[1], 1.0, 1e-5, "rosenbrock_num y=1");
    printf("  iterations=%d, lambda_final=%.2e\n",
           res.iterations, res.lambda_final);
}

/* ================================================================
 * 测试 4: 指数拟合 y = a*exp(b*x) + c
 *
 * 已知参数: a=2.0, b=-0.5, c=1.0
 * 合成 8 个数据点 (含轻微噪声, 使残差不为精确零)
 * ================================================================ */

typedef struct { double t[8]; double y[8]; } exp_data_t;

static void exp_residual(const double *p, void *data,
                          double *r, int m, int n)
{
    (void)n;
    exp_data_t *d = (exp_data_t *)data;
    double a = p[0], b = p[1], c = p[2];
    for (int i = 0; i < m; i++)
        r[i] = a * exp(b * d->t[i]) + c - d->y[i];
}

static void exp_jacobian(const double *p, void *data,
                          double *J, int m, int n)
{
    (void)n;
    exp_data_t *d = (exp_data_t *)data;
    double a = p[0], b = p[1];
    for (int i = 0; i < m; i++) {
        double ebx = exp(b * d->t[i]);
        J[i * 3 + 0] = ebx;              /* ∂r/∂a */
        J[i * 3 + 1] = a * d->t[i] * ebx; /* ∂r/∂b */
        J[i * 3 + 2] = 1.0;              /* ∂r/∂c */
    }
}

static void test_exp_fit(void)
{
    printf("exponential fit (analytic J) ...\n");

    exp_data_t d;
    int m = 8, n = 3;
    for (int i = 0; i < m; i++) {
        d.t[i] = (double)i * 0.5;   /* t = 0, 0.5, 1.0, ..., 3.5 */
        d.y[i] = 2.0 * exp(-0.5 * d.t[i]) + 1.0;
    }

    double x[] = {1.0, -0.2, 0.5};  /* 偏离真值的初值 */
    lm_result_t res;
    int ret = lm_solve(x, m, n, exp_residual, exp_jacobian,
                       &d, NULL, &res);

    CHECK(ret == 0, "exp converged");
    CHECK_NEAR(x[0], 2.0, 1e-4, "exp a=2");
    CHECK_NEAR(x[1], -0.5, 1e-4, "exp b=-0.5");
    CHECK_NEAR(x[2], 1.0, 1e-4, "exp c=1");
    CHECK_NEAR(res.final_cost, 0.0, 1e-12, "exp cost≈0");
    printf("  iterations=%d, lambda_final=%.2e\n",
           res.iterations, res.lambda_final);
}

/* ================================================================
 * 测试 5: 默认配置 NULL 参数
 * ================================================================ */

static void test_null_config(void)
{
    printf("NULL config / result ...\n");

    double x[] = {-1.2, 1.0};
    int ret = lm_solve(x, 2, 2, rosenbrock_residual, rosenbrock_jacobian,
                       NULL, NULL, NULL);
    CHECK(ret == 0, "NULL config converged");
    CHECK_NEAR(x[0], 1.0, 1e-6, "NULL config x=1");
}

/* ================================================================
 * 测试 6: 参数边界检查
 * ================================================================ */

static void test_bounds(void)
{
    printf("bounds check ...\n");

    lm_result_t res;
    double x[4];
    /* m < n 应失败 */
    int ret = lm_solve(x, 2, 4, NULL, NULL, NULL, NULL, &res);
    CHECK(ret == -2, "m < n → -2");

    /* n > LM_MAX_PARAMS 应失败 */
    double xbig[LM_MAX_PARAMS + 1];
    ret = lm_solve(xbig, LM_MAX_RESIDUALS, LM_MAX_PARAMS + 1,
                   NULL, NULL, NULL, NULL, &res);
    CHECK(ret == -2, "n > MAX → -2");

    /* NULL residual 应失败 */
    ret = lm_solve(x, 3, 2, NULL, NULL, NULL, NULL, &res);
    CHECK(ret == -2, "NULL residual → -2");
}

/* ================================================================ */

int main(void)
{
    test_linear_ls();
    test_rosenbrock_analytic();
    test_rosenbrock_numerical();
    test_exp_fit();
    test_null_config();
    test_bounds();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
