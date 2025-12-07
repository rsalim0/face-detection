#!/usr/bin/env python3
import cv2
import time
import numpy as np
import sys
import serial

# -------------------------- CONFIG --------------------------
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'
CAM_IDX = 0
FRAME_SIZE = (800, 600)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# --- Motor tuning ---
MAX_ROTATE_STEP = 10       # max degrees per command
THRESHOLD = 20             # minimum pixels offset to start correction
SMOOTH_FACTOR = 0.04       # smaller = smoother (0.03–0.1 is good)
SEND_INTERVAL = 0.05        # seconds between commands
# -------------------------------------------------------------

def main():
    # ---- Connect to Arduino ----
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[OK] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print("[ERROR] Could not connect to Arduino:", e)
        arduino = None

    # ---- Load cascade ----
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        sys.exit(f"[ERROR] Could not load cascade: {CASCADE_PATH}")

    # ---- Open camera ----
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open camera")

    frame_center_x = FRAME_SIZE[0] // 2
    last_send = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            cx = x + w // 2
            offset = cx - frame_center_x

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, y + h // 2), 5, (0, 0, 255), -1)

            # ---- Proportional smooth rotation ----
            if abs(offset) > THRESHOLD and (time.time() - last_send > SEND_INTERVAL):
                # Map offset to rotation angle (proportional control)
                rotate_deg = np.clip(abs(offset) * SMOOTH_FACTOR, 1, MAX_ROTATE_STEP)

                if offset < 0:
                    command = f"CCW {rotate_deg:.1f}\n"
                    print(f"Left ({offset}) → CCW {rotate_deg:.1f}°")
                else:
                    command = f"CW {rotate_deg:.1f}\n"
                    print(f"Right ({offset}) → CW {rotate_deg:.1f}°")

                if arduino:
                    arduino.write(command.encode('utf-8'))
                last_send = time.time()

        # Draw center line
        # cv2.line(frame, (frame_center_x, 0), (frame_center_x, FRAME_SIZE[1]), (255, 0, 0), 2)

        cv2.imshow("Smooth Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()
