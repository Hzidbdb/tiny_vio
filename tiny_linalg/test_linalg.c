#include <stdio.h>
#include <math.h>
#include "tiny_linalg.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("  FAIL [%d]: %s\n", __LINE__, msg); } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(fabs((a)-(b)) < (tol), msg)

static void test_vec_dot(void) {
    printf("vec_dot ...\n");
    double a[] = {1,2,3,4,5};
    double b[] = {2,3,4,5,6};
    CHECK_NEAR(vec_dot(a,b,5), 70.0, 1e-10, "dot product 5");
    CHECK_NEAR(vec_dot(a,b,3), 20.0, 1e-10, "dot product 3 (tail)");
}

static void test_vec_add(void) {
    printf("vec_add / vec_remove ...\n");
    double a[]={1,2,3}, b[]={4,5,6}, out[3];
    vec_add(a,b,out,3);
    CHECK_NEAR(out[0],5,1e-10,"add[0]");
    CHECK_NEAR(out[2],9,1e-10,"add[2]");
    vec_remove(a,b,out,3);
    CHECK_NEAR(out[0],-3,1e-10,"sub[0]");
}

static void test_vec_cross(void) {
    printf("vec_cross ...\n");
    double a[]={1,0,0}, b[]={0,1,0}, out[3];
    vec_cross(a,b,out);
    CHECK_NEAR(out[2],1.0,1e-10,"cross Z");
}

static void test_vec_L2(void) {
    printf("vec_L2 ...\n");
    double a[]={3,4};
    CHECK_NEAR(vec_L2(a,2),5.0,1e-10,"L2 norm");
}

static void test_mat_vec_mul(void) {
    printf("mat_vec_mul ...\n");
    double A[]={1,2, 3,4, 5,6}; // 3x2
    double x[]={10,20}, y[3];
    mat_vec_mul(A,x,y,3,2);
    CHECK_NEAR(y[0],50,1e-10,"mv[0]");   // 1*10+2*20
    CHECK_NEAR(y[2],170,1e-10,"mv[2]");  // 5*10+6*20
}

static void test_mat_mul(void) {
    printf("mat_mul ...\n");
    double A[]={1,2,3, 4,5,6};    // 2x3
    double B[]={1,2, 3,4, 5,6};  // 3x2
    double C[4];
    mat_mul(A,B,C,2,3,2);
    CHECK_NEAR(C[0],22,1e-10,"mul[0]");  // 1*1+2*3+3*5
    CHECK_NEAR(C[3],64,1e-10,"mul[3]");  // 4*2+5*4+6*6
}

static void test_mat_transpose(void) {
    printf("mat_transpose ...\n");
    double A[]={1,2,3, 4,5,6}, B[6];
    mat_transpose(A,B,2,3);
    CHECK_NEAR(B[0],1,1e-10,"T[0]");
    CHECK_NEAR(B[2],2,1e-10,"T[2]");
}

static void test_mat_add_sub(void) {
    printf("mat_add / mat_sub ...\n");
    double A[]={1,2,3}, B[]={4,5,6}, C[3];
    mat_add(A,B,C,1,3);
    CHECK_NEAR(C[1],7,1e-10,"add");
    mat_sub(A,B,C,1,3);
    CHECK_NEAR(C[2],-3,1e-10,"sub");
}

static void test_mat_vec_madd(void) {
    printf("mat_vec_madd ...\n");
    double A[]={1,2, 3,4, 5,6};
    double x[]={10,20}, y[]={1,2,3}, out[3];
    mat_vec_madd(A,x,y,out,3,2);
    CHECK_NEAR(out[0],51,1e-10,"madd[0]");
}

static void test_mat_rank1_update(void) {
    printf("mat_rank1_update ...\n");
    double A[]={1,0, 0,1};
    double x[]={2,3};
    mat_rank1_update(A,x,1.0,2);
    CHECK_NEAR(A[0],5,1e-10,"r1[0]"); // 1 + 2*2
    CHECK_NEAR(A[3],10,1e-10,"r1[3]"); // 1 + 3*3
}

static void test_mat_solve(void) {
    printf("mat_solve ...\n");
    double A[]={2,1, 1,3};
    double b[]={5,6}, x[2];
    double Acopy[4];
    mat_copy(A,Acopy,2,2);
    int ret = mat_solve(Acopy,x,b,2);
    CHECK(ret==0,"solve ret");
    CHECK_NEAR(x[0],1.8,1e-10,"x0");  // x=1.8, y=1.4
    CHECK_NEAR(x[1],1.4,1e-10,"x1");
}

static void test_mat_cholesky(void) {
    printf("mat_cholesky ...\n");
    double A[]={4,2, 2,3};  // [4 2; 2 3], pos-def
    double L[4], x[2], b[]={8,7};
    int ret = mat_cholesky(A,L,2);
    CHECK(ret==0,"chol ret");
    // L = [2 0; 1 sqrt(2)]
    CHECK_NEAR(L[0],2.0,1e-10,"L00");
    CHECK_NEAR(L[2],1.0,1e-10,"L10");
    mat_cholesky_solve(L,b,x,2);
    // Ax=b -> [4 2;2 3][x0;x1]=[8;7] -> x0=1.25, x1=1.5
    CHECK_NEAR(x[0],1.25,1e-10,"chol x0");
    CHECK_NEAR(x[1],1.5,1e-10,"chol x1");
}

static void test_mat33_symeig(void) {
    printf("mat33_symeig ...\n");
    // diag(3,2,1)
    double A[]={3,0,0, 0,2,0, 0,0,1};
    double lambda[3], V[9];
    mat33_symeig(A,lambda,V);
    CHECK_NEAR(lambda[0],3,1e-10,"eig[0]");
    CHECK_NEAR(lambda[1],2,1e-10,"eig[1]");
    CHECK_NEAR(lambda[2],1,1e-10,"eig[2]");
    // V should be I (up to sign)
    CHECK(fabs(fabs(V[0])-1.0)<1e-10,"V00");
}

static void test_mat33_svd(void) {
    printf("mat33_svd ...\n");
    // diag(3,2,1) -> SVD: U=I, S=[3,2,1], V=I
    double A[]={3,0,0, 0,2,0, 0,0,1};
    double U[9], S[3], V[9];
    mat33_svd(A,U,S,V);
    CHECK_NEAR(S[0],3,1e-10,"svd S0");
    CHECK_NEAR(S[1],2,1e-10,"svd S1");
    CHECK_NEAR(S[2],1,1e-10,"svd S2");
    // Reconstruct: U*diag(S)*V^T should equal A
    double US[9], VT[9], recon[9];
    for(int i=0;i<9;i++) US[i]=U[i];
    for(int j=0;j<3;j++) for(int i=0;i<3;i++) US[i*3+j]*=S[j];
    mat_transpose(V,VT,3,3);
    mat_mul(US,VT,recon,3,3,3);
    for(int i=0;i<9;i++)
        CHECK_NEAR(recon[i],A[i],1e-8,"svd recon");
}

static void test_mat33_exp_so3(void) {
    printf("mat33_exp_so3 ...\n");
    // zero rotation
    double w0[]={0,0,0}, R[9];
    mat33_exp_so3(w0,R);
    for(int i=0;i<9;i++) {
        double e = (i%4==0)?1.0:0.0;
        CHECK_NEAR(R[i],e,1e-10,"exp0");
    }
    // 90 deg around Z
    double w90[]={0,0,M_PI/2};
    mat33_exp_so3(w90,R);
    CHECK_NEAR(R[0],0,1e-10,"exp90 R00");
    CHECK_NEAR(R[1],-1,1e-10,"exp90 R01");
    CHECK_NEAR(R[3],1,1e-10,"exp90 R10");
}

static void test_mat33_log_so3(void) {
    printf("mat33_log_so3 ...\n");
    // log(exp(w)) = w
    double w_in[]={0.5,-0.3,0.8}, R[9], w_out[3];
    mat33_exp_so3(w_in,R);
    mat33_log_so3(R,w_out);
    CHECK_NEAR(w_out[0],w_in[0],1e-10,"log0");
    CHECK_NEAR(w_out[1],w_in[1],1e-10,"log1");
    CHECK_NEAR(w_out[2],w_in[2],1e-10,"log2");

    // pi rotation (edge case)
    double w_pi[]={M_PI,0,0}, Rp[9], wp[3];
    mat33_exp_so3(w_pi,Rp);
    mat33_log_so3(Rp,wp);
    CHECK_NEAR(fabs(wp[0]),M_PI,1e-6,"log pi");
}

static void test_mat_power_inv(void) {
    printf("mat_power_inv ...\n");
    // 3x3 diag(5,3,1), smallest eigenvalue is 1
    double A[]={5,0,0, 0,3,0, 0,0,1};
    double v[]={1,1,1}, work[9];
    double lam = mat_power_inv(A,v,work,3,0.0,20,1e-10);
    CHECK_NEAR(lam,1.0,1e-8,"pow_inv lambda");
}

static void test_utility(void) {
    printf("utility ...\n");
    double A[]={1,2, 3,4}, C[4];
    mat_copy(A,C,2,2);
    CHECK_NEAR(C[0],1,1e-10,"copy");
    double I[9];
    mat_identity(I,3);
    CHECK_NEAR(I[0],1,1e-10,"I00");
    CHECK_NEAR(I[1],0,1e-10,"I01");
    CHECK_NEAR(I[4],1,1e-10,"I11");
    double diag[2];
    mat_get_diag(A,diag,2);
    CHECK_NEAR(diag[0],1,1e-10,"get_diag1");
    CHECK_NEAR(diag[1],4,1e-10,"get_diag2");
}

int main(void) {
    test_vec_dot();
    test_vec_add();
    test_vec_cross();
    test_vec_L2();
    test_mat_vec_mul();
    test_mat_mul();
    test_mat_transpose();
    test_mat_add_sub();
    test_mat_vec_madd();
    test_mat_rank1_update();
    test_mat_solve();
    test_mat_cholesky();
    test_mat33_symeig();
    test_mat33_svd();
    test_mat33_exp_so3();
    test_mat33_log_so3();
    test_mat_power_inv();
    test_utility();
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
