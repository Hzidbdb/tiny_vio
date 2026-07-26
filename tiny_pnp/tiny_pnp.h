#ifndef TINY_PNP_H
#define TINY_PNP_H

/* ============================================================
 * tiny_pnp — EPnP 位姿解算
 *
 * 纯 C99, 依赖 tiny_linalg + tiny_opt, 栈分配, 无 malloc。
 * 给定 n 个 2D-3D 对应点 + 相机内参 → 输出相机位姿 R,t。
 *
 * 算法: EPnP (Lepetit et al. 2009) + Horn 绝对定向 + LM 重投影精化
 * ============================================================ */

#define PNP_MAX_POINTS  80

/* ---------- 相机内参 ---------- */

typedef struct {
    double fx, fy;      /* 焦距 (像素)         */
    double cx, cy;      /* 主点 (像素)         */
} pnp_camera_t;

/* ---------- 结果 ---------- */

/* P_cam = R * P_world + t (世界 → 相机) */
typedef struct {
    double R[9];        /* 旋转矩阵 3×3 (行优先)  */
    double t[3];        /* 平移向量               */
    int    iterations;  /* LM 精化迭代次数         */
    double reproj_err;  /* 最终平均重投影误差(像素)*/
} pnp_result_t;

/* ---------- API ---------- */

/* 主求解器
 * points_3D: n×3 世界坐标 (行优先)
 * points_2D: n×2 像素坐标 (行优先)
 * n:   点数 (4 ≤ n ≤ PNP_MAX_POINTS)
 * cam: 相机内参 (不可为 NULL)
 * 返回: 0 成功, -1 失败 (点数不足/退化)  */
int pnp_solve(const double *points_3D, const double *points_2D,
              int n, const pnp_camera_t *cam,
              pnp_result_t *result);

#endif /* TINY_PNP_H */
