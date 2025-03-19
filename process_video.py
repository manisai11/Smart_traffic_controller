import numpy as np
import cv2
from collections import deque

next_id = 0
centroid_tracker = {}
TRACK_MEMORY = 10

def load_class_names():
    with open("coco.names", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

def process_video(file_path):
    global next_id
    cap = cv2.VideoCapture(file_path)
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        current_centroids = []
        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        indices = indices.flatten() if indices is not None and len(indices) > 0 else []

        for i in indices:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])

            if label in ["car", "bus", "truck", "motorbike", "bicycle"]:
                cx, cy = x + w // 2, y + h // 2
                current_centroids.append((cx, cy))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (cx, cy) in current_centroids:
            match_found = False
            for vehicle_id, centroids in list(centroid_tracker.items()):
                if np.linalg.norm(np.array(centroids[-1]) - np.array((cx, cy))) < 30:
                    centroid_tracker[vehicle_id].append((cx, cy))
                    match_found = True
                    break
            
            if not match_found:
                centroid_tracker[next_id] = deque([(cx, cy)], maxlen=TRACK_MEMORY)
                next_id += 1
                vehicle_count += 1

    cap.release()
    return vehicle_count
