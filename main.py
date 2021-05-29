import time

import cv2
import numpy as np
from keras_preprocessing.image import img_to_array
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

import box_utils_numpy


def predict(
    width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1
):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils_numpy.hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4],
    )


options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
options.add_session_config_entry("session.load_model_format", "ORT")
face_model = InferenceSession("model/slim-facedetect.ort", options)
smoke_model = InferenceSession("model/smoke-detect.onnx")
face_inputname = face_model.get_inputs()[0].name
smoke_inputname = smoke_model.get_inputs()[0].name
print("Model loaded")

cap = cv2.VideoCapture("smoking.mp4")
image_mean = np.array([127, 127, 127])
prev_frame_time = 0
threshold = 0.7

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    confidences, boxes = face_model.run(None, {face_inputname: image})
    boxes, labels, probs = predict(
        orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold
    )

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Preprocessing for smoke detection model (frame)
    gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    img_gray = np.zeros_like(orig_image)
    img_gray[:, :, 0] = gray
    img_gray[:, :, 1] = gray
    img_gray[:, :, 2] = gray

    for i in range(boxes.shape[0]):
        (startX, startY, endX, endY) = boxes[i, :]
        # Preprocessing for smoke detection model (face)
        face = img_gray[startX:endX, startY:endY]
        try:
            face = cv2.resize(face, (32, 32))
        except cv2.error as e:
            continue
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        predicted = smoke_model.run(None, {smoke_inputname: face})
        ids = np.argmax(predicted)
        label = "Smoking" if bool(ids) else "Not Smoking"

        cv2.putText(
            orig_image,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        cv2.rectangle(orig_image, (startX, startY), (endX, endY), (255, 255, 0), 4)
    
    fps = str(round(fps))
    # puting the FPS count on the frame
    cv2.putText(
        orig_image,
        fps,
        (0, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (100, 255, 0),
        3,
        cv2.LINE_AA,
    )

    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow(f"SMOKING_DETECTOR", orig_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
