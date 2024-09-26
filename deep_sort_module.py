import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort  # Make sure to install the deep_sort_realtime package


class DeepSORT:
    def __init__(self):
        self.deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

    def process(self, frame, detections):
        bbox_xywh = np.array([d[:4] for d in detections])
        confidences = np.array([d[4] for d in detections])

        tracks = self.deepsort.update_tracks(bbox_xywh, confidences, frame)

        output_detections = []
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()  # Get bounding box in top-left bottom-right format
            output_detections.append((track_id, *bbox))

        return output_detections


def get_deep_sort():
    return DeepSORT()
