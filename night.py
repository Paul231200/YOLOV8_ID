import gc
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Настройка параметров
stream_url = 0
tracker_config = 'bytetrack.yaml'

# Инициализация модели
model = YOLO("yolov8n.pt")

# Цвета для боксов
COLORS = np.random.uniform(0, 255, size=(80, 3))

def draw_boxes(image, boxes, scores, labels, ids, class_names):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i] % len(COLORS)]
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[labels[i]]
        label = f'ID: {ids[i] if ids[i] is not None else "No ID"} {class_name}: {scores[i]:.2f}'

        # Рисование прямоугольника
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Вычисление положения текста
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Рисование региона интереса (ROI)
        roi_x1 = max(x1 - 10, 0)
        roi_y1 = max(y1 - 10, 0)
        roi_x2 = min(x2 + 10, image.shape[1])
        roi_y2 = min(y2 + 10, image.shape[0])
        roi_label = f'ROI (X: {roi_x1}, Y: {roi_y1})'
        
        cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)  # Синий цвет, толстая рамка
        (w, h), _ = cv2.getTextSize(roi_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (roi_x2, roi_y1 - h - 10), (roi_x2 + w, roi_y1), (255, 0, 0), -1)
        cv2.putText(image, roi_label, (roi_x2, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_fps(image, fps):
    fps_text = f'FPS: {fps:.2f}'
    (w, h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (image.shape[1] - w - 10, 10), (image.shape[1], 10 + h + 10), (0, 0, 0), -1)
    cv2.putText(image, fps_text, (image.shape[1] - w - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_static_roi(image):
    h, w = image.shape[:2]
    overlay = image.copy()
    alpha = 0.3  # Прозрачность заливки
    roi_x1, roi_y1 = w // 2, 0
    roi_x2, roi_y2 = w, h

    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    roi_label = 'Static ROI'
    (label_w, label_h), _ = cv2.getTextSize(roi_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (roi_x1 + 10, roi_y1 + 10), (roi_x1 + 10 + label_w, roi_y1 + 10 + label_h), (255, 0, 0), -1)
    cv2.putText(image, roi_label, (roi_x1 + 10, roi_y1 + 10 + label_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def is_in_roi(box, roi_x1, roi_y1, roi_x2, roi_y2):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    intersection_x1 = max(x1, roi_x1)
    intersection_y1 = max(y1, roi_y1)
    intersection_x2 = min(x2, roi_x2)
    intersection_y2 = min(y2, roi_y2)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    return intersection_area > 0.5 * box_area

try:
    # Попытка начать отслеживание
    results = model.track(source=stream_url, show=False, tracker=tracker_config, stream=True)
    class_names = model.names  # Получение названий классов из модели

    prev_time = time.time()
    for r in results:
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        image = r.orig_img
        h, w = image.shape[:2]
        roi_x1, roi_y1 = w // 2, 0
        roi_x2, roi_y2 = w, h

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([None] * len(boxes))

        for i in range(len(boxes)):
            if not is_in_roi(boxes[i], roi_x1, roi_y1, roi_x2, roi_y2):
                ids[i] = None  # Обнуление ID, если объект не находится в статическом ROI
        
        # Рисование боксов
        draw_boxes(image, boxes, scores, labels, ids, class_names)
        
        # Рисование статического региона интереса (ROI)
        draw_static_roi(image)
        
        # Отображение FPS
        draw_fps(image, fps)
        
        # Отображение изображения
        cv2.imshow("YOLOv8 Tracking", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except ConnectionError as e:
    print(f"Ошибка подключения: {e}")
except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    del results
    del model
    gc.collect()
    cv2.destroyAllWindows()
