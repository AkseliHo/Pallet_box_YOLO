"""
RealSense-kameran live-näkymä ja kuvien tallennus.

Ohjelman toiminta:
- Pyytää käyttäjältä tallennuskansion ja tarkistaa sen olemassaolon.
- Käynnistää RealSense-pipelinen ilman manuaalisia asetuksia; SDK valitsee
  automaattisesti käytettävät streamit.
- Kohdistaa syvyysdatan värikuvan koordinaatistoon (rs.align).
- Näyttää värikuvan ja värillisen syvyyskartan vierekkäin.
- Painamalla 'S' tallentaa:
    * värikuvan (PNG)
    * syvyyskartan (PNG)
    * syvyysdatan raakamuodossa (NPY)
- Painamalla ESC ohjelma sulkeutuu.

"""


import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# Kansion määrittely
save_dir = input("Anna tallennuskansion polku (esim. C:/data/realsense): ").strip()
if not os.path.isdir(save_dir):
    raise FileNotFoundError(f"Kansiota ei löydy: {save_dir}")

# Luo pipeline ja konfiguraatio
pipeline = rs.pipeline()
config = rs.config()

# Käynnistä ilman manuaalisia asetuksia – SDK valitsee automaattisesti tuetut streamit
profile = pipeline.start()

# Tulosta laitteen nimi ja streamit
device = profile.get_device()
print(f"\nKäytössä oleva laite: {device.get_info(rs.camera_info.name)}")
print("[Käyttöohjeet] S = tallenna kuva, ESC = lopeta\n")

# Kohdistus väriin
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Syvyyskartta värillisenä
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense Live", combined)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            color_path = os.path.join(save_dir, f"color_{timestamp}.png")
            depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")
            depth_raw_path = os.path.join(save_dir, f"depth_raw_{timestamp}.npy")

            cv2.imwrite(color_path, color_image)
            cv2.imwrite(depth_path, depth_colormap)
            np.save(depth_raw_path, depth_image)

            print(f"Tallennettu:\n  {color_path}\n  {depth_path}\n  {depth_raw_path}")
        elif key == 27:
            print("Suljetaan...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
