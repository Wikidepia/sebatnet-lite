import cv2
import numpy as np
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

import box_utils_numpy


class SebatNet:
    def __init__(self, tvm=False):
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.tvm = tvm
        if self.tvm:
            self.load_tvm()
        else:
            self.face_model = InferenceSession("model/slim-facedetect.onnx", options)
            self.face_inputname = self.face_model.get_inputs()[0].name
        self.smoke_model = InferenceSession("model/smoke-detect.onnx", options)
        self.smoke_inputname = self.smoke_model.get_inputs()[0].name

    def load_tvm(self):
        import tvm
        from tvm.contrib import graph_executor

        device = tvm.cpu()
        fd_lib = tvm.runtime.load_module("model/version-slim-320.so")
        self.face_model = graph_executor.GraphModule(fd_lib["default"](device))

    def find_boxes(
        self,
        width,
        height,
        confidences,
        boxes,
        prob_threshold,
        iou_threshold=0.3,
        top_k=-1,
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

    def find_smokers(self, img):
        ret = []
        threshold = 0.7
        image_mean = np.array([127, 127, 127])

        image = cv2.resize(img, (320, 240))
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        if self.tvm:
            self.face_model.run(input=image)
            confidences = self.face_model.get_output(0).asnumpy()
            boxes = self.face_model.get_output(1).asnumpy()
        else:
            confidences, boxes = self.face_model.run(None, {self.face_inputname: image})
        boxes, _, _ = self.find_boxes(
            img.shape[1], img.shape[0], confidences, boxes, threshold
        )

        # Preprocessing for smoke detection model (frame)
        gray = img[..., :3] @ [0.299, 0.587, 0.114]
        gray = gray.astype("float") / 255.0
        gray = np.asarray(gray, dtype="float32")
        for i in range(boxes.shape[0]):
            (startX, startY, endX, endY) = boxes[i, :]

            # Preprocessing for smoke detection model (face)
            face = gray[startY:endY, startX:endX]
            if face.size == 0:  # Clamping x, y is probably better
                continue
            face = cv2.resize(face, (32, 32))
            face = face.reshape((face.shape[0], face.shape[1], 1))
            face = np.expand_dims(face, axis=0)
            predicted = self.smoke_model.run(None, {self.smoke_inputname: face})
            predicted_id = np.argmax(predicted)
            ret.append(
                {
                    "coords": (startX, startY, endX, endY),
                    "is_smoking": bool(predicted_id),
                }
            )
        return ret
