import cv2
import numpy as np


def draw_star(img: np.ndarray, center: np.ndarray, size: int, color: tuple, thickness: int) -> None:
    points = []
    for i in range(5):
        angle = i * 144  # 144 degree: angle between each vertex of the five-pointed star
        x = int(center[0] + size * np.sin(np.radians(angle)))
        y = int(center[1] + size * np.cos(np.radians(angle)))
        points.append((x, y))
    points = np.array(points, np.int32)

    cv2.polylines(
        img=img, 
        pts=[points], 
        isClosed=True, 
        color=color, 
        thickness=thickness
    )
