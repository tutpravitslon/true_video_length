import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple
from datetime import datetime

class ImageDateTimeClassifier:
    def __init__(self, model_path: str, plate_bbox_relative: Tuple[float, float, float, float],
                 digit_positions: List[float], digit_width: float, date_format: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape[:1:-1])  # [b, 1, h, w] -> [w, h]
        self.plate_bbox = plate_bbox_relative
        self.rel_digit_positions = np.array(digit_positions)
        self.rel_digit_width = digit_width
        self.date_format = date_format

    def get_plate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]  # высота / ширина 
        nhs, nws, nhe, nwe = self.plate_bbox
        return image[round(h * nhs):round(h * nhe), round(w * nws):round(w * nwe)]

    def get_crops(self, plate: np.ndarray) -> List[np.ndarray]:
        plate_width = plate.shape[1]
        digit_starts = plate_width * self.rel_digit_positions
        digit_width = plate_width * self.rel_digit_width
        return [plate[:, round(s):round(s + digit_width)] for s in digit_starts]


    def classify_digits(self, crops: List[np.ndarray]) -> np.ndarray:
      
        crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crops]
        crops = [cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for crop in crops]

        crops = np.stack([cv2.resize(crop, self.input_shape)[None] for crop in crops])
        ort_inputs = {self.input_name: crops.astype(np.float32)}

        pred_logits = self.session.run(None, ort_inputs)[0]
        predicted_digit = pred_logits.argmax(axis=-1)
        return predicted_digit

    def parse_timestamp(self, image: np.ndarray) -> int:
        plate = self.get_plate(image)
        crops = self.get_crops(plate)
        digit_predictions = self.classify_digits(crops) # [1 9 0 6 2 0 2 4 1 9 0 9 5 2]
        timestamp_str = "".join(digit_predictions.astype(str)) # 19062024190937
        timestamp = int(datetime.strptime(timestamp_str, self.date_format).timestamp()) #1718813392
        # print(datetime.strptime(timestamp_str, self.date_format))  # 2024-06-19 19:09:52
        # print(datetime.strptime(timestamp_str, self.date_format).timestamp()) # 1718813392.0
        return timestamp
