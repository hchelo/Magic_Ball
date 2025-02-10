import os
import platform
import sys
from pathlib import Path
import torch
import time
import numpy as np
import cv2

frame_count = 0  # Inicializa el contador de cuadros

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, scale_boxes, non_max_suppression)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights='./yolov9s.pt',
        source=0,  # Usar la cámara por defecto (0)
        data=ROOT / 'data/coco.yaml',
        imgsz=(640, 640),
        conf_thres=0.4,
        iou_thres=0.4,
        max_det=1000,
        device='',
        classes=[32],  # Especificar la clase a detectar
        agnostic_nms=False,
        augment=False,
        line_thickness=3,
):
    frame_count = 0  # Inicializa el contador de cuadros
    frame_skip = 1  # Procesar 1 de cada 10 cuadros

    # Cargar el modelo
    device = select_device(device)
    LOGGER.info(f"Using device: {device}")
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Inicializar la cámara
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        LOGGER.error("Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1
        start_time = time.time()

        # Preprocesamiento
        h, w, _ = frame.shape
        scale = min(imgsz[0] / h, imgsz[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        im_resized = cv2.resize(frame, (new_w, new_h))
        padded_im = 255 * np.ones((imgsz[0], imgsz[1], 3), dtype=np.uint8)
        top, left = (imgsz[0] - new_h) // 2, (imgsz[1] - new_w) // 2
        padded_im[top:top + new_h, left:left + new_w] = im_resized
        im = padded_im[..., ::-1].copy().astype(float) / 255.0
        im = torch.from_numpy(im).to(model.device).permute(2, 0, 1).unsqueeze(0).float()

        # Inferencia
        pred = model(im, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)

        # Dibujar detecciones
        detected = False
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[-2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if not detected:
                        cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                        radius = int(min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) / 2)
                        cv2.putText(frame, f"({cx}, {cy}), Radio: {radius}", (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)
                        detected = True

                        if radius >= 10:
                            text = "SALTAR"
                            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
                            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                            text_x, text_y = (frame.shape[1] - text_size[0]) // 2, 50
                            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        # Mostrar el cuadro procesado
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
