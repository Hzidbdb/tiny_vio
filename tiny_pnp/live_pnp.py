#!/usr/bin/env python3
"""
实时摄像头 PnP 位姿测算

支持两种模式:
  1. 本地窗口: python3 live_pnp.py
  2. HTTP 流 (SSH远程): python3 live_pnp.py --http 8080
     浏览器打开 http://<ip>:8080 即可实时观看

用法:
    python3 live_pnp.py --help

棋盘格要求:
    打印一张 7×10 内角点棋盘格, 方格边长 25mm
    贴在硬板上, 确保相机能看到完整棋盘格
"""

import cv2
import numpy as np
import ctypes
import argparse
import time
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ================================================================
# C 库接口
# ================================================================

class Camera(ctypes.Structure):
    _fields_ = [("fx", ctypes.c_double), ("fy", ctypes.c_double),
                ("cx", ctypes.c_double), ("cy", ctypes.c_double)]

class PnpResult(ctypes.Structure):
    _fields_ = [("R", ctypes.c_double * 9), ("t", ctypes.c_double * 3),
                ("iterations", ctypes.c_int), ("reproj_err", ctypes.c_double)]

_libdir = os.path.dirname(os.path.abspath(__file__))
_so = ctypes.CDLL(os.path.join(_libdir, "libtinypnp.so"))
_so.pnp_solve.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.POINTER(Camera), ctypes.POINTER(PnpResult)]
_so.pnp_solve.restype = ctypes.c_int

def pnp_solve(pts3d, pts2d, fx, fy, cx, cy):
    n = len(pts3d)
    p3 = (ctypes.c_double * (n*3))(*pts3d.ravel())
    p2 = (ctypes.c_double * (n*2))(*pts2d.ravel())
    cam = Camera(fx, fy, cx, cy)
    res = PnpResult()
    if _so.pnp_solve(p3, p2, n, cam, res) != 0:
        return None
    R = np.array(list(res.R)).reshape(3, 3)
    t = np.array(list(res.t))
    return R, t, res.iterations, res.reproj_err

# ================================================================
# 棋盘格
# ================================================================

def make_checkerboard(rows, cols, square_mm):
    pts = np.zeros((rows * cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            pts[r*cols + c] = [c * square_mm, r * square_mm, 0.0]
    return pts

# ================================================================
# 绘制辅助
# ================================================================

def draw_pose(frame, R, t, fx, fy, cx, cy, reproj_err, iterations):
    """在帧上绘制坐标轴和位姿信息"""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # 坐标轴 (红X 绿Y 蓝Z)
    axis_len = 50.0  # mm
    axis_3d = np.array([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]], dtype=np.float64)
    axis_2d = cv2.projectPoints(axis_3d, cv2.Rodrigues(R)[0], t, K, None)[0].reshape(-1,2).astype(int)
    origin = tuple(axis_2d[0])
    cv2.line(frame, origin, tuple(axis_2d[1]), (0,0,255), 2)
    cv2.line(frame, origin, tuple(axis_2d[2]), (0,255,0), 2)
    cv2.line(frame, origin, tuple(axis_2d[3]), (255,0,0), 2)

    # 3D 方框 (棋盘格范围)
    p3d_box = np.array([[0,0,0],[100,0,0],[100,100,0],[0,100,0]], dtype=np.float64)
    p2d_box = cv2.projectPoints(p3d_box, cv2.Rodrigues(R)[0], t, K, None)[0].reshape(-1,2).astype(int)
    cv2.polylines(frame, [p2d_box], True, (0,255,255), 2)

    # 文字信息
    pitch = np.arctan2(-R[2,0], np.hypot(R[2,1], R[2,2]))
    yaw   = np.arctan2(R[1,0], R[0,0])
    dist  = np.linalg.norm(t)
    info = [
        f"Pos: X={t[0]:6.1f} Y={t[1]:6.1f} Z={t[2]:6.1f} mm",
        f"Dist={dist:.0f}mm  Pitch={np.degrees(pitch):+5.1f}deg  Yaw={np.degrees(yaw):+5.1f}deg",
        f"Reproj={reproj_err:.3f}px  LM={iterations}",
    ]
    for li, line in enumerate(info):
        cv2.putText(frame, line, (10, frame.shape[0]-40+20*li),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ================================================================
# HTTP MJPG 流 (SSH 无 GUI 时用浏览器查看)
# ================================================================

class PnPStreamer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def set_frame(self, jpg):
        with self.lock:
            self.frame = jpg

    def get_frame(self):
        with self.lock:
            return self.frame

g_streamer = PnPStreamer()

class MJPGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """<!DOCTYPE html>
<html><head><title>PnP Live</title><meta charset="utf-8">
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style></head>
<body><img src="/stream"></body></html>"""
            self.wfile.write(html.encode())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    jpg = g_streamer.get_frame()
                    if jpg is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='实时摄像头 PnP 位姿测算')
    parser.add_argument('--cam', type=int, default=0, help='摄像头ID')
    parser.add_argument('--http', type=int, default=0, help='HTTP流端口 (0=本地窗口)')
    parser.add_argument('--rows', type=int, default=7, help='棋盘格行数')
    parser.add_argument('--cols', type=int, default=10, help='棋盘格列数')
    parser.add_argument('--square', type=float, default=25.0, help='方格mm')
    parser.add_argument('--fx', type=float, default=450, help='fx')
    parser.add_argument('--fy', type=float, default=450, help='fy')
    parser.add_argument('--cx', type=float, default=320, help='cx')
    parser.add_argument('--cy', type=float, default=240, help='cy')
    args = parser.parse_args()

    # HTTP 服务器
    if args.http > 0:
        server = HTTPServer(('0.0.0.0', args.http), MJPGHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"HTTP 流: http://<本机IP>:{args.http}/")
        print(f"本地浏览器打开, 或 wget http://localhost:{args.http}/stream 测试\n")

    # 棋盘格 3D 点
    pts3d = make_checkerboard(args.rows, args.cols, args.square)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 摄像头
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"摄像头 {args.cam} 打不开"); sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头: {w}×{h}")
    print(f"棋盘格: {args.rows}×{args.cols}={len(pts3d)} 角点, {args.square}mm 方格")
    print(f"内参: fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    print(f"按 'q' 退出, 's' 截图\n")

    fps_times = []
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
            cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners2, found)

            result = pnp_solve(pts3d, corners2.reshape(-1,2),
                               args.fx, args.fy, args.cx, args.cy)
            if result:
                R, t, iters, rerr = result
                draw_pose(frame, R, t, args.fx, args.fy, args.cx, args.cy, rerr, iters)

        # FPS
        fps_times.append(time.time() - t0)
        if len(fps_times) > 30: fps_times.pop(0)
        fps = 1.0 / (sum(fps_times)/len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"FPS:{fps:.0f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 编码 JPEG
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            g_streamer.set_frame(jpg.tobytes())

        if args.http == 0:
            cv2.imshow("PnP Live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                cv2.imwrite(time.strftime("pnp_%Y%m%d_%H%M%S.png"), frame)
                print("截图已保存")

    cap.release()
    if args.http == 0:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
