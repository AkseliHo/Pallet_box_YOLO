from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import pyrealsense2 as rs
import socket
import time

# =========================================================
# PARAMETERS
# =========================================================
N_SAMPLES = 10
TOTAL_TIME_SEC = 2.0

POS_TOL_MM = 10.0       # mm (X ja Y)
Z_TOL_MM = 40.0         # mm (depth)
ANGLE_TOL_DEG = 3.0     # deg

MAX_RETRIES = 20          # Out of tolerances --> how many times try to retry
RETRY_DELAY_SEC = 2   # delay between retrys

DEBUG_MEASUREMENTS = True

# =========================================================
# CUDA
# =========================================================
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
print(f"{'CUDA in use:' if cuda_available else 'CPU in use:'} {device_name}")

# =========================================================
# TCP SERVER (Python = server)
# =========================================================
SERVER_IP = "<tupe ip here>"
SERVER_PORT = # <Type server port>

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(1)

# =========================================================
# RealSense
# =========================================================
class RealSenseCamera:
    def __init__(self):
        serial = " TYPE HERE SERIAL NUMBER"
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print("RealSense ready.")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        depth_raw = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_raw * depth_scale  # meter

        # Realsense settings 0–3 m
        min_depth = 0
        max_depth = 3.0

        depth_vis = np.clip((depth_m - min_depth) / (max_depth - min_depth) * 255, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        return color, depth_m, depth_colormap

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense closed.")

# =========================================================
# YOLO
# =========================================================
PROJECT_DIR = "<type here project directory>"
IMG_SIZE = 960
best_model_path = os.path.join(PROJECT_DIR, "< type here best.pt file location and file name>")
model = YOLO(best_model_path)
cam = RealSenseCamera()

# =========================================================
# DEBUG: deviation printing
# =========================================================
def print_unstable_debug(xs, ys, zs, angles):
    print("[DEBUG] UNSTABLE_TARGET detected")
    print(f"  X values: {['%.1f' % x for x in xs]} | ΔX = {max(xs)-min(xs):.1f} mm")
    print(f"  Y values: {['%.1f' % y for y in ys]} | ΔY = {max(ys)-min(ys):.1f} mm")
    print(f"  Z values: {['%.1f' % z for z in zs]} | ΔZ = {max(zs)-min(zs):.1f} mm")
    print(f"  A values: {['%.2f' % a for a in angles]} | ΔA = {max(angles)-min(angles):.2f} deg")

# =========================================================
# COORDINATE MEASUREMENT
# =========================================================
def single_measurement():
    color, depth_raw, depth_colormap = cam.get_frames()
    if color is None:
        return None

    results = model.predict(color, imgsz=IMG_SIZE, task="segment", verbose=False)[0]
    if results.masks is None:
        return None

    candidates = []

    for mask, conf in zip(results.masks.xy, results.boxes.conf):
        poly = np.array(mask, dtype=np.int32)
        M = cv2.moments(poly)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        depth_val = depth_raw[cy, cx] if 0 <= cy < depth_raw.shape[0] and 0 <= cx < depth_raw.shape[1] else 0
        if depth_val <= 0:
            continue

        rect = cv2.minAreaRect(poly)
        angle = rect[2] + 90 if rect[1][0] < rect[1][1] else rect[2]

        X, Y, Z = rs.rs2_deproject_pixel_to_point(cam.intrinsics, [cx, cy], depth_val)

        # Area
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)
        edge1 = np.linalg.norm(box_pts[0]-box_pts[1])
        edge2 = np.linalg.norm(box_pts[1]-box_pts[2])
        area_m2 = (edge1*depth_val/600) * (edge2*depth_val/600) if depth_val>0 else 0

        # Box size class that is sent to ABB robot
        size_class = 1 if area_m2 > 0.12 else 2  # 1 = H, 2 = S

        candidates.append((X*1000, Y*1000, Z*1000, angle, depth_val, float(conf), size_class))

    if not candidates:
        return None

    # Nearest box to camera is chosen (smallest Z)
    candidates.sort(key=lambda c: c[4])
    return candidates[0][:7]



# =========================================================
# VALIDATE CAPTURES
# =========================================================
def capture_and_validate():
    samples = []
    delay = TOTAL_TIME_SEC / N_SAMPLES

    for _ in range(N_SAMPLES):
        meas = single_measurement()
        if meas is not None:
            samples.append(meas)
        time.sleep(delay)

    if len(samples) < N_SAMPLES:
        return None

    xs, ys, zs, angles, _, _, sizes = zip(*samples)  # sizes = kaikki mittausten koko-luokat

    # size_class is set what value has been detected the most in a single try
    size_class_majority = max(set(sizes), key=sizes.count)

    if DEBUG_MEASUREMENTS:
        print("[DEBUG] All measurements:")
        for i, (x, y, z, a, _, _, sz) in enumerate(samples):
            print(f"  Sample {i+1}: X={x:.1f}, Y={y:.1f}, Z={z:.1f}, A={a:.2f}, Size={sz}")
        print(f"[DEBUG] Majority class size: {size_class_majority}")

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    da = max(angles) - min(angles)

    if dx > POS_TOL_MM or dy > POS_TOL_MM or dz > Z_TOL_MM or da > ANGLE_TOL_DEG:
        print("[ERROR] UNSTABLE_TARGET – nothing is being sent to ABB")
        if DEBUG_MEASUREMENTS:
            print_unstable_debug(xs, ys, zs, angles)
            print(f"  Tolerances: POS={POS_TOL_MM} mm, Z={Z_TOL_MM} mm, ANGLE={ANGLE_TOL_DEG} deg")
        return None

    X = np.mean(xs) - 10
    Y = np.mean(ys) + 150
    Z = np.mean(zs)
    A = np.mean(angles) + 88

    # COORDINATES ARE SENT TO ABB
    return f"{X:.1f};{Y:.1f};{Z:.1f};{A:.2f};{size_class_majority}\n"


# =========================================================
# REALTIME VISUALISATION - USER INTERFACE
# =========================================================
def run_realtime_visualization():
    print("[INFO] Real time visualization. ESC = end")

    while True:
        color, depth_raw, depth_map = cam.get_frames()
        if color is None:
            continue

        img = color.copy()
        results = model.predict(img, imgsz=IMG_SIZE, task="segment", verbose=False)[0]

        if results.masks is not None:
            boxes_list = []

            for mask_points, box, cls_id, conf in zip(
                results.masks.xy, results.boxes.xyxy,
                results.boxes.cls, results.boxes.conf
            ):
                polygon = np.array(mask_points, dtype=np.int32)
                cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

                M = cv2.moments(polygon)
                cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

                depth_val = depth_raw[cy, cx] if 0 <= cy < depth_raw.shape[0] and 0 <= cx < depth_raw.shape[1] else 0
                if depth_val > 0:
                    cv2.putText(img, f"Z: {depth_val:.3f} m", (cx + 10, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                rect = cv2.minAreaRect(polygon)
                box_pts = cv2.boxPoints(rect)
                box_pts = np.int32(box_pts)
                cv2.polylines(img, [box_pts], True, (255, 255, 0), 2)

                edge1 = np.linalg.norm(box_pts[0]-box_pts[1])
                edge2 = np.linalg.norm(box_pts[1]-box_pts[2])
                dx, dy = (box_pts[1][0]-box_pts[0][0], box_pts[1][1]-box_pts[0][1]) if edge1>=edge2 else (box_pts[2][0]-box_pts[1][0], box_pts[2][1]-box_pts[1][1])
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0: angle += 360
                cv2.putText(img, f"Rot: {angle:.1f}°", (cx+10, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                area_m2 = (edge1*depth_val/600) * (edge2*depth_val/600) if depth_val>0 else 0
                size_class = 1 if area_m2 > 0.12 else 2  # sama kuin ABB:lle
                size_str = "H" if size_class == 1 else "S"  # näytölle
                cv2.putText(img, f"Area: {area_m2:.3f} m2", (cx+10, cy+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(img, f"Size: {size_str}", (cx+10, cy+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


                boxes_list.append((cx, cy, area_m2, depth_val))

            if boxes_list:
                boxes_list.sort(key=lambda x: (x[3], -x[2]))
                cx1, cy1, _, _ = boxes_list[0]
                cv2.putText(img, "1st", (cx1, cy1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        live = np.hstack((img, depth_map))
        cv2.imshow("Live YOLO + Depth + All Attributes", live)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# =========================================================
# MAIN LOOP
# =========================================================
try:
    import threading

    vis_thread = threading.Thread(target=run_realtime_visualization, daemon=True)
    vis_thread.start()

    print("[INFO] Waiting connection to ABB...")

    while True:
        print("[INFO] Waiting ABB-connection in port number ", SERVER_PORT)
        conn, addr = server_socket.accept()
        print(f"[INFO] ABB connected: {addr}")

        try:
            while True:
                cmd_raw = conn.recv(1024)
                if not cmd_raw:
                    print("[WARN] Connection failure, waiting for new connection...")
                    break

                cmd = cmd_raw.decode("utf-8").strip()
                print(f"[RX ABB] {cmd}")

                if cmd == "CAPTURE":
                    print("[INFO] CAPTURE requested")
                    for attempt in range(1, MAX_RETRIES + 1):
                        reply = capture_and_validate()
                        if reply is not None:
                            print(f"[TX ABB] {reply.strip()}")
                            conn.sendall(reply.encode("utf-8"))
                            break
                        else:
                            print(f"[WARN] Error in measurement, attempt number {attempt}/{MAX_RETRIES}")
                            if attempt < MAX_RETRIES:
                                time.sleep(RETRY_DELAY_SEC)
                            else:
                                error_msg = "ERROR,MAX_RETRIES_EXCEEDED\n"
                                print(f"[TX ABB] {error_msg.strip()}")
                                conn.sendall(error_msg.encode("utf-8"))

                elif cmd == "EXIT":
                    print("[INFO] EXIT received, Exiting program...")
                    raise KeyboardInterrupt

                else:
                    print(f"[WARN] Unknown command: {cmd}")

        finally:
            conn.close()
            print("[INFO] Connection closed, waiting for a new ABB connection...")

except KeyboardInterrupt:
    print("[INFO] Ending program by request...")

finally:
    cam.stop()
    server_socket.close()
    print("[INFO] Closed succesfully.")
