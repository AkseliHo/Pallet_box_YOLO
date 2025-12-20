from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np
import torch
import pyrealsense2 as rs

# ---------- TARKISTA CUDA KERRAN ALKUKSI ----------
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
print(f"{'CUDA käytössä, GPU löytyy:' if cuda_available else 'CUDA ei käytössä, käytetään CPU:'} {device_name}")
# -----------------------------------

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start()
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        depth_raw = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_raw, alpha=0.03),
            cv2.COLORMAP_JET
        )

        return color, depth_raw, depth_colormap

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


# ---------- LAATIKON KOKOJA VOI MUUTTAA TÄSTÄ ----------
# Koot metreinä: leveys x pituus
BOX_SIZES = {
    "small": (0.15, 0.30),   # 15 cm x 30 cm
    "large": (0.40, 1.00),   # 40 cm x 100 cm
    # Lisättävissä uusia kokoja: "medium": (0.25, 0.50)
}
# ---------------------------------------------------------

# ---------- ASETUKSET ----------
DATA_YAML = "dataset/data.yaml"
MODEL_ARCH = "yolov8n-seg.pt"  # segmentation malli!
EPOCHS = 1800
IMG_SIZE = 960
PROJECT_DIR = "runs/train_results"
TEST_IMAGES_DIR = "dataset/valid/images"   # RGB-kuvat -kansio
DEPTH_RAW_DIR = "dataset/valid/depth_raw"  # depth_raw_*.npy -kuvat kansio
RESULTS_OUT_DIR = "runs/infer_results"
# --------------------------------


def train_model():
    """Kouluttaa YOLOv8-seg mallin."""
    # --- Ladataan malli ja ilmoitetaan tyyppi heti ---
    model = YOLO(MODEL_ARCH)
    try:
        model_name = model.model.yaml.get("name", "unknown")
        print(f"Loaded model type: {model_name}")
    except:
        print("Could not read model type from weights.")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        cache=True,      # välimuisti GPU:lle, VAIHDA disk tarvittaessa, nyt on ram
        half=True,      # FP16, pienentää muistinkäyttöä ja nopeuttaa
        project=PROJECT_DIR,
        name="pallet_boxes",
        exist_ok=True,
        task="segment",
        verbose = False  # <- poistaa ylimääräiset cuda käytössä -tulostukset
    )
    print(f"\nKoulutus valmis! Malli tallennettu kansioon: {PROJECT_DIR}/pallet_boxes")


def run_realtime_camera_inference():
    """Reaaliaikainen YOLO-segmentointi RealSense-kameralla

    """
    best_model_path = os.path.join(PROJECT_DIR, "pallet_boxes", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Mallia ei löydy. Kouluta malli ensin.")

    model = YOLO(best_model_path)
    cam = RealSenseCamera()
    print("\n[Käyttöohje]\nESC = lopeta\n")

    try:
        while True:
            color, depth_raw, depth_map = cam.get_frames()
            if color is None:
                continue

            img = color.copy()
            h, w = img.shape[:2]

            results = model.predict(img, imgsz=IMG_SIZE, task="segment", verbose=False)[0]

            if results.masks is not None:
                for mask_points, box, cls_id, conf in zip(
                    results.masks.xy, results.boxes.xyxy,
                    results.boxes.cls, results.boxes.conf
                ):
                    polygon = np.array(mask_points, dtype=np.int32)
                    cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

                    # Keskipiste
                    M = cv2.moments(polygon)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
                    else:
                        cx = cy = 0

                    # Syvyys
                    depth_val = None
                    if 0 <= cy < depth_raw.shape[0] and 0 <= cx < depth_raw.shape[1]:
                        depth_val = depth_raw[cy, cx]
                        if depth_val > 10:
                            depth_val = depth_val / 1000.0  # mm → m

                    # MinAreaRect ja rotaatio
                    rect = cv2.minAreaRect(polygon)
                    box_pts = cv2.boxPoints(rect)
                    box_pts = np.int32(box_pts)
                    cv2.polylines(img, [box_pts], True, (255, 255, 0), 2)
                    angle = rect[2] + 90 if rect[1][0] < rect[1][1] else rect[2]

                    cv2.putText(img, f"Rot: {angle:.1f}°", (cx + 10, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if depth_val:
                        cv2.putText(img, f"Z: {depth_val:.2f} m", (cx + 10, cy + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # ---------- Laske laatikon sivut ja pinta-ala ----------
                    if depth_val is not None:
                        pixel_width = np.linalg.norm(box_pts[0] - box_pts[1])
                        pixel_height = np.linalg.norm(box_pts[1] - box_pts[2])
                        F = 600     # SKAALUS, Säädä tarvittaessa
                        width_m = pixel_width * depth_val / F
                        height_m = pixel_height * depth_val / F
                        area_m2 = width_m * height_m

                        cv2.putText(img, f"Area: {area_m2:.3f} m2",
                                    (cx + 10, cy + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Näytä live-kuva
            live = np.hstack((img, depth_map))
            cv2.imshow("Live YOLO + Depth + Area (ESC = quit)", live)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.stop()


def load_depth_for_image(img_name):
    """
    Palauttaa syvyyskartan (depth_raw .npy) vastaavalle RGB-kuvalle.
    Tukee Roboflowin uudelleennimeämiä tiedostoja (esim. .rf.<hash>.jpg).
    """
    base = os.path.basename(img_name)
    name_no_ext = os.path.splitext(base)[0]

    # Poistetaan mahdollinen Roboflowin lisäosa (esim. "_png.rf.<hash>")
    # ja palautetaan alkuperäinen aikaleimaosa
    # esim: color_20251107_154639_png.rf.asdasd -> 20251107_154639
    name_clean = name_no_ext.replace("color_", "")
    if "_png" in name_clean:
        name_clean = name_clean.split("_png")[0]
    elif "_jpg" in name_clean:
        name_clean = name_clean.split("_jpg")[0]

    # Rakennetaan depth-tiedoston nimi
    depth_file = f"depth_raw_{name_clean}.npy"
    depth_path = os.path.join(DEPTH_RAW_DIR, depth_file)

    # Jos ei löydy, yritetään vielä osittaista matchia
    if not os.path.exists(depth_path):
        import glob
        candidates = glob.glob(os.path.join(DEPTH_RAW_DIR, f"depth_raw*{name_clean}*.npy"))
        if candidates:
            depth_path = candidates[0]
        else:
            print(f" Depth-tiedostoa ei löytynyt!!!: {depth_path}")
            return None

    depth = np.load(depth_path)

    # Skaalataan jos eri kokoinen kuin RGB
    rgb_path = os.path.join(TEST_IMAGES_DIR, img_name)
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is not None:
        h_rgb, w_rgb = rgb_img.shape[:2]
        if depth.shape[:2] != (h_rgb, w_rgb):
            print(f"Depth-kuvan koko {depth.shape[::-1]} ≠ RGB {w_rgb}x{h_rgb}, skaalaan...")
            depth = cv2.resize(depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

    return depth


def run_inference():
    """Suorittaa inferenssin ja lisää laatikon keskipisteen etäisyyden sekä pinta-alan syvyysdatan perusteella."""
    best_model_path = os.path.join(PROJECT_DIR, "pallet_boxes", "weights",
                                   "best.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            "Mallia ei löytynyt. Aja ensin koulutus (train_model()).")

    model = YOLO(best_model_path)
    os.makedirs(RESULTS_OUT_DIR, exist_ok=True)
    results_data = []

    for img_name in os.listdir(TEST_IMAGES_DIR):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        depth_map = load_depth_for_image(img_name)
        results = \
        model.predict(img, imgsz=IMG_SIZE, task="segment", verbose=False)[0]

        if results.masks is not None:
            for mask_points, box, cls_id, conf in zip(
                    results.masks.xy, results.boxes.xyxy, results.boxes.cls,
                    results.boxes.conf
            ):
                polygon = np.array(mask_points, dtype=np.int32)
                cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

                # Keskipiste
                M = cv2.moments(polygon)
                cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

                # Syvyys
                depth_val = None
                if depth_map is not None and 0 <= cy < depth_map.shape[
                    0] and 0 <= cx < depth_map.shape[1]:
                    depth_val = depth_map[cy, cx]
                    if depth_val > 10:
                        depth_val = depth_val / 1000.0  # mm → m

                # Vinon laatikon rotaatio
                rect = cv2.minAreaRect(polygon)
                box_pts = cv2.boxPoints(rect)
                box_pts = np.int32(box_pts)
                cv2.polylines(img, [box_pts], True, (255, 255, 0), 2)
                angle = rect[2] + 90 if rect[1][0] < rect[1][1] else rect[2]

                # Piirretään labelit
                cv2.putText(img, f"{model.names[int(cls_id)]} {conf:.2f}",
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if depth_val:
                    cv2.putText(img, f"Z: {depth_val:.2f} m",
                                (cx + 5, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                                2)

                # ---------- Pinta-alan laskenta ----------
                if depth_val is not None:
                    pixel_width = np.linalg.norm(box_pts[0] - box_pts[1])
                    pixel_height = np.linalg.norm(box_pts[1] - box_pts[2])
                    F = 600  # karkea fokaalipisteen arvio
                    width_m = pixel_width * depth_val / F
                    height_m = pixel_height * depth_val / F
                    area_m2 = width_m * height_m
                    cv2.putText(img, f"Area: {area_m2:.3f} m2",
                                (cx + 5, cy + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                                2)

        # Tallenna kuva
        out_path = os.path.join(RESULTS_OUT_DIR, img_name)
        cv2.imwrite(out_path, img)
        print(f"Tallennettu: {out_path}")


if __name__ == "__main__":

    # ----Valitse toiminto----

    DO_TRAIN = False     
    DO_INFER = False
    DO_REALTIME = True   # LIVE YOLO

    #-------------------------#

    if DO_TRAIN:
        train_model()
    if DO_INFER:
        run_inference()
    if DO_REALTIME:
        run_realtime_camera_inference()
