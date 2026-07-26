/* ============================================================
 * live_pnp.cpp — 实时摄像头 PnP 位姿测算 (C++ OpenCV 薄封装)
 *
 * 用 OpenCV C++ API 做摄像头采集 + 棋盘格检测 + 显示，
 * PnP 解算调用 tiny_pnp (纯C) 库。
 *
 * 编译 (从项目根目录):
 *   g++ -std=c++17 -O2 -o tiny_pnp/live_pnp tiny_pnp/live_pnp.cpp \
 *       -Ltiny_pnp -ltinypnp $(pkg-config --cflags --libs opencv4) \
 *       -Wl,-rpath,'$ORIGIN'
 *
 * 用法:
 *   ./tiny_pnp/live_pnp [--cam 0] [--rows 7] [--cols 10] [--square 25] ...
 * ============================================================ */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
#include <string>

extern "C" {
#include "tiny_pnp.h"
}

/* ---- 棋盘格 3D 点 ---- */
static std::vector<cv::Point3f> make_chessboard(int rows, int cols, float square_mm)
{
    std::vector<cv::Point3f> pts;
    pts.reserve(rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            pts.push_back(cv::Point3f(c * square_mm, r * square_mm, 0.0f));
    return pts;
}

/* ---- 绘制位姿坐标轴和信息 ---- */
static void draw_pose(cv::Mat &frame,
                      const double R[9], const double t[3],
                      double fx, double fy, double cx, double cy,
                      double reproj_err, int iterations)
{
    cv::Mat Rmat(3, 3, CV_64F, const_cast<double*>(R));
    cv::Mat tvec(3, 1, CV_64F, const_cast<double*>(t));
    cv::Mat rvec;
    cv::Rodrigues(Rmat, rvec);

    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    /* 坐标轴: 红X 绿Y 蓝Z */
    std::vector<cv::Point3f> axis3d = {
        {0,0,0}, {50,0,0}, {0,50,0}, {0,0,50}
    };
    std::vector<cv::Point2f> axis2d;
    cv::projectPoints(axis3d, rvec, tvec, K, cv::noArray(), axis2d);
    cv::Point o(axis2d[0].x, axis2d[0].y);
    cv::line(frame, o, cv::Point(axis2d[1].x, axis2d[1].y), cv::Scalar(0,0,255), 2);
    cv::line(frame, o, cv::Point(axis2d[2].x, axis2d[2].y), cv::Scalar(0,255,0), 2);
    cv::line(frame, o, cv::Point(axis2d[3].x, axis2d[3].y), cv::Scalar(255,0,0), 2);

    /* 棋盘格边框 (黄色) */
    std::vector<cv::Point3f> box3d = {
        {0,0,0}, {100,0,0}, {100,100,0}, {0,100,0}
    };
    std::vector<cv::Point2f> box2d;
    cv::projectPoints(box3d, rvec, tvec, K, cv::noArray(), box2d);
    std::vector<cv::Point> box_pts;
    for (auto &p : box2d) box_pts.push_back(cv::Point((int)p.x, (int)p.y));
    cv::polylines(frame, box_pts, true, cv::Scalar(0,255,255), 2);

    /* 文字信息 */
    double pitch = std::atan2(-R[6], std::hypot(R[7], R[8]));
    double yaw   = std::atan2(R[3], R[0]);
    double dist  = std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);

    char buf[128];
    int y0 = frame.rows - 40;
    std::snprintf(buf, sizeof(buf), "Pos: X=%6.1f Y=%6.1f Z=%6.1f mm", t[0], t[1], t[2]);
    cv::putText(frame, buf, cv::Point(10, y0),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    std::snprintf(buf, sizeof(buf), "Dist=%.0fmm  Pitch=%+5.1fdeg  Yaw=%+5.1fdeg",
                  dist, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
    cv::putText(frame, buf, cv::Point(10, y0 + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    std::snprintf(buf, sizeof(buf), "Reproj=%.3fpx  LM=%d", reproj_err, iterations);
    cv::putText(frame, buf, cv::Point(10, y0 + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
}

/* ---- 主函数 ---- */
int main(int argc, char **argv)
{
    int    rows = 7, cols = 10;
    float  square_mm = 25.0f;
    double fx = 450, fy = 450, cx = 320, cy = 240;
    int    cam_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "--cam"   || a == "--rows" || a == "--cols" ||
             a == "--fx"    || a == "--fy"   || a == "--cx" ||
             a == "--cy"    || a == "--square") && i+1 < argc) {
            std::string v = argv[++i];
            if      (a == "--cam")    cam_id    = std::stoi(v);
            else if (a == "--rows")   rows      = std::stoi(v);
            else if (a == "--cols")   cols      = std::stoi(v);
            else if (a == "--square") square_mm = std::stof(v);
            else if (a == "--fx")     fx        = std::stod(v);
            else if (a == "--fy")     fy        = std::stod(v);
            else if (a == "--cx")     cx        = std::stod(v);
            else if (a == "--cy")     cy        = std::stod(v);
        }
    }

    /* 棋盘格 3D 点 */
    auto pts3d = make_chessboard(rows, cols, square_mm);
    cv::Size pattern(cols, rows);
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    /* 摄像头 */
    cv::VideoCapture cap(cam_id);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "Cannot open camera %d\n", cam_id);
        return 1;
    }

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::printf("Camera: %dx%d\n", w, h);
    std::printf("Chessboard: %dx%d=%zu corners, %.1fmm square\n",
                rows, cols, pts3d.size(), square_mm);
    std::printf("Intrinsics: fx=%.0f fy=%.0f cx=%.0f cy=%.0f\n", fx, fy, cx, cy);
    std::printf("Press 'q' to quit, 's' to screenshot\n\n");

    pnp_camera_t pnp_cam = {fx, fy, cx, cy};
    cv::Mat frame, gray;
    std::vector<cv::Point2f> corners;
    std::vector<double> fps_times;

    while (true) {
        int64 t0 = cv::getTickCount();
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        bool found = cv::findChessboardCorners(gray, pattern, corners);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(5,5), cv::Size(-1,-1), criteria);
            cv::drawChessboardCorners(frame, pattern, corners, found);

            std::vector<double> p3d(pts3d.size() * 3);
            std::vector<double> p2d(corners.size() * 2);
            for (size_t i = 0; i < pts3d.size(); i++) {
                p3d[i*3+0] = pts3d[i].x;
                p3d[i*3+1] = pts3d[i].y;
                p3d[i*3+2] = pts3d[i].z;
            }
            for (size_t i = 0; i < corners.size(); i++) {
                p2d[i*2+0] = corners[i].x;
                p2d[i*2+1] = corners[i].y;
            }

            pnp_result_t result;
            int ret = pnp_solve(p3d.data(), p2d.data(), (int)corners.size(),
                                &pnp_cam, &result);
            if (ret == 0) {
                draw_pose(frame, result.R, result.t,
                          fx, fy, cx, cy,
                          result.reproj_err, result.iterations);
            } else {
                /* PnP 失败，显示红色警告 */
                cv::putText(frame, "PnP FAILED (ret=" + std::to_string(ret) + ")",
                            cv::Point(10, frame.rows - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
            }
        }

        /* FPS */
        double dt = (cv::getTickCount() - t0) / cv::getTickFrequency();
        fps_times.push_back(dt);
        if (fps_times.size() > 30) fps_times.erase(fps_times.begin());
        double avg = 0;
        for (double t : fps_times) avg += t;
        avg /= fps_times.size();
        char fps_buf[32];
        std::snprintf(fps_buf, sizeof(fps_buf), "FPS:%.0f", 1.0 / avg);
        cv::putText(frame, fps_buf, cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);

        cv::imshow("PnP Live", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 's') {
            std::time_t now = std::time(nullptr);
            char fname[64];
            std::strftime(fname, sizeof(fname), "pnp_%Y%m%d_%H%M%S.png",
                          std::localtime(&now));
            cv::imwrite(fname, frame);
            std::printf("Screenshot saved: %s\n", fname);
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
