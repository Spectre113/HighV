import cv2
import numpy as np
from controller import Camera
from typing import Dict


def detect_yellow_lane_and_error(image):
    """
    Detect yellow lane and compute angle deviation from vertical center.

    Args:
        image: RGB numpy array

    Returns:
        tuple: (processed_image, error) or (original_image, 0.0) if detection fails
    """
    if image is None or image.size == 0:
        return image, 0.0

    try:
        # 1. Convert to HSV for better yellow color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define yellow color range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # 2. Create yellow mask
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 3. Edge detection on yellow regions
        edges = cv2.Canny(yellow_mask, 250, 252)

        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20
        )

        if lines is None:
            return image, 0.0

        # 5. Extract center coordinates
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle and center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            filtered_lines.append((x1, y1, x2, y2, center_x, center_y))

        # 5. Group lines by proximity and select dominant one
        lines_by_center = {}
        for line in filtered_lines:
            x1, y1, x2, y2, center_x, center_y = line

            # Find if this line is close to existing group
            found_group = False
            for existing_center in lines_by_center.keys():
                if (
                    abs(center_x - existing_center) < image.shape[1] * 0.1
                ):  # 10% width threshold
                    lines_by_center[existing_center].append(line)
                    found_group = True
                    break

            if not found_group:
                lines_by_center[center_x] = [line]

        # Select group with most lines (dominant lane)
        if not lines_by_center:
            return image, 0.0

        dominant_lines = filtered_lines

        # 6. Average the dominant lines into one representative line
        avg_x1 = np.mean([l[0] for l in dominant_lines])
        avg_y1 = np.mean([l[1] for l in dominant_lines])
        avg_x2 = np.mean([l[2] for l in dominant_lines])
        avg_y2 = np.mean([l[3] for l in dominant_lines])
        avg_center_x = np.mean([l[4] for l in dominant_lines])
        avg_center_y = np.mean([l[5] for l in dominant_lines])

        # 7. Draw the detected line on original image
        result_image = image.copy()

        cv2.line(
            result_image,
            (int(avg_x1), int(avg_y1)),
            (int(avg_x2), int(avg_y2)),
            (255, 0, 0),
            2,
        )

        cv2.circle(
            result_image,
            (int(avg_center_x), int(avg_center_y)),
            3,
            (0, 255, 0),
            3,
        )

        # Draw vertical reference line
        center_x = image.shape[1] // 2
        cv2.line(
            result_image, (center_x, 0), (center_x, image.shape[0]), (255, 0, 0), 2
        )  # Blue reference line

        w = image.shape[1]
        lateral_error = (avg_center_x - w // 2) / (w // 2)

        # Add angle text
        angle_text = f"Error: {lateral_error:.1f}"
        cv2.putText(
            result_image,
            angle_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return result_image, lateral_error

    except Exception as e:
        print(f"Error in lane detection: {e}")
        return image, 0.0


class MultiCameralineDetector:

    def __init__(self):
        self.current_camera = 'central'

    def __call__(self, cameras: Dict[str, Camera]):

        images = {}
        for camera_name, camera in cameras.items():
            img = self.read_img(camera)
            images[camera_name] = img

        max_abs_error = -np.inf

        result_image = None
        result_error = 0.0

        central_lane, central_error = detect_yellow_lane_and_error(images['central'])
        if central_error > 0:
            self.current_camera = 'central'
            return central_lane, central_error

        left_lane, left_error = detect_yellow_lane_and_error(images['left'])
        right_lane, right_error = detect_yellow_lane_and_error(images['right'])

        if self.current_camera == 'left':
            return left_lane, left_error
        elif self.current_camera == 'right':
            return right_lane, right_error

        if np.abs(left_error) > np.abs(right_error):
            self.current_camera = 'left'
            return left_lane, left_error

        self.current_camera = 'right'
        return right_lane, right_error

    def read_img(self, camera: Camera):
        width = camera.getWidth()
        height = camera.getHeight()
        img_bytes = camera.getImage()
        img = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 4)).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img


class BaselineDetector:

    def __call__(self, camera: Camera):
        img = self.read_img(camera)
        return detect_yellow_lane_and_error(img)

    def read_img(self, camera: Camera):
        width = camera.getWidth()
        height = camera.getHeight()
        img_bytes = camera.getImage()
        img = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 4)).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img
