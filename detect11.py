import os
import platform
import sys
from pathlib import Path
import torch
import time
import numpy as np

frame_count = 0  # Inicializa el contador de cuadros

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights='./yolov9s.pt',
        source='videos/balonazo.mp4',
        data=ROOT / 'data/coco.yaml',
        imgsz=(640, 640),
        conf_thres=0.4,  # Ajustar el umbral de confianza para ser más estrictos
        iou_thres=0.4,   # Ajustar el umbral de supresión para eliminar detecciones solapadas
        max_det=1000,
        device='',
        classes=[32],  # Especificar la clase que se desea detectar
        agnostic_nms=False,
        augment=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
):
    frame_count = 0  # Inicializa el contador de cuadros
    frame_skip = 1  # Procesar 1 de cada 10 cuadros

    source = str(source)
    is_file = Path(source).suffix[1:] in ['mp4', 'avi', 'mov']
    if not is_file:
        LOGGER.error(f"Invalid source: {source}. Must be a valid file path.")
        return

    # Obtener el nombre del archivo sin la extensión y agregar el prefijo "output_"
    video_name = Path(source).stem  # Obtiene el nombre del archivo sin la extensión
    output_path = f"output_{video_name}.mp4"  # Añade el prefijo 'output_' al nombre del archivo

    # Cargar el modelo
    device = select_device(device)
    LOGGER.info(f"Using device: {device}")
    model = DetectMultiBackend(weights, device=device, data=data)  # Definir el modelo aquí
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Obtener propiedades del video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        LOGGER.error(f"Cannot open video: {source}. Check the file path or format.")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usar el codec MP4
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtener los cuadros por segundo del video original
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crear el objeto VideoWriter para guardar el video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar solo 1 de cada 10 cuadros
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1
        start_time = time.time()

        # Preprocesar la imagen
        h, w, _ = frame.shape
        # Redimensionar la imagen manteniendo la relación de aspecto
        scale = min(imgsz[0] / h, imgsz[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        im_resized = cv2.resize(frame, (new_w, new_h))
        
        # Crear una imagen de fondo de tamaño imgsz y colocar la imagen redimensionada en el centro
        top = (imgsz[0] - new_h) // 2
        left = (imgsz[1] - new_w) // 2
        padded_im = 255 * np.ones(shape=[imgsz[0], imgsz[1], 3], dtype=np.uint8)
        padded_im[top:top+new_h, left:left+new_w] = im_resized

        # Convertir de BGR a RGB
        im = padded_im[..., ::-1]

        # Crear una copia para evitar el problema de los strides negativos
        im = im.copy()

        # Normalizar la imagen (valores entre 0 y 1)
        im = im.astype(float) / 255.0

        # Convertir la imagen a tensor de PyTorch
        im = torch.from_numpy(im).to(model.device).permute(2, 0, 1).unsqueeze(0).float()

        # Realizar la inferencia
        pred = model(im, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)

        # Log predictions
        LOGGER.info(f"Predictions: {pred}")

        # Procesar predicciones y mostrar solo un objeto
        detected = False
        for det in pred:
            if len(det):
                # Aplicar la supresión de máximos
                det[:, :4] = scale_boxes(im.shape[-2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if not detected:
                        cx = int((xyxy[0] + xyxy[2]) / 2)
                        cy = int((xyxy[1] + xyxy[3]) / 2)
                        
                        radius = int(min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) / 2)  # Radio basado en el menor lado
                        # Crear el texto con las coordenadas y el radio
                        coordinates_text = f"({cx}, {cy}), Radio: {radius}"
                        # Dibujar el texto con las coordenadas y el radio en el cuadro
                        
                        # Dibujar el rectángulo
                        #cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                        detected = True  # Solo dibujar el primer objeto detectado
                        # Dibujar el círculo en lugar de un rectángulo
                        cv2.putText(frame, coordinates_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)  # Color azul, grosor 2

                        if radius >= 10:
                            # Dibujar el texto "SALTAR" en la parte superior central de la imagen
                            text = "SALTAR"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5  # Aumenta el tamaño de la fuente
                            font_thickness = 3  # Grosor de las letras
                            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                            text_x = (frame.shape[1] - text_size[0]) // 2  # Centrar horizontalmente
                            text_y = 50  # Ubicación en la parte superior
                            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)


        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        LOGGER.info(f"FPS: {fps:.2f}")

        # Guardar el cuadro con las detecciones en el archivo de salida
        out.write(frame)

        # Mostrar el cuadro procesado en pantalla
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Liberar el VideoWriter
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
