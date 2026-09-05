import cv2
import numpy as np


def check_point_in_stop_zone(point, polygon):
    """
    Check whether a point is inside a polygon.
    """
    x, y = point
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def get_side_of_line(point, line_point1, line_point2):
    """
    Determine which side of line a point is on.
    Returns positive/negative value.
    """
    px, py = point
    ax, ay = line_point1
    bx, by = line_point2

    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def check_if_exit_store(previous, current, entrance_a, entrance_b):
    previous_side = get_side_of_line(previous, entrance_a, entrance_b)
    current_side = get_side_of_line(current, entrance_a, entrance_b)

    if previous_side > 0 and current_side < 0:
        return True
    return False


def check_if_enter_store(previous, current, entrance_a, entrance_b):
    """
    Determine whether a trajectory crossed a line and toward direction of entering store
    """
    previous_side = get_side_of_line(previous, entrance_a, entrance_b)
    current_side = get_side_of_line(current, entrance_a, entrance_b)

    if previous_side < 0 and current_side > 0:
        return True
    return False


def put_top_right_text(text, image_height, image_width, image, height_margin=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    color = (0, 255, 150)  # Green in BGR
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Set top-right coordinates with a 20-pixel margin from the edges
    margin = 20
    x = image_width - text_width - margin
    y = (
        text_height + margin + height_margin
    )  # Y flows downward, so add height to pull it into view

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def detect_face(face_model, person_img):
    face_results = face_model(person_img, conf=0.4, verbose=False)

    if len(face_results) < 1 or len(face_results[0].boxes) < 1:
        return None

    face_box = face_results[0].boxes[0]
    face_x1, face_y1, face_x2, face_y2 = face_box.xyxy[0].cpu().numpy().astype(int)

    pad_x = int((face_x2 - face_x1) * 0.2)
    pad_y = int((face_y2 - face_y1) * 0.2)
    px1 = max(0, face_x1 - pad_x)
    py1 = max(0, face_y1 - pad_y)
    px2 = min(person_img.shape[1], face_x2 + pad_x)
    py2 = min(person_img.shape[0], face_y2 + pad_y)

    face_crop = person_img[py1:py2, px1:px2]
    return face_crop


def detect_head_pose(pose_model, face_img):
    """
    Return: only yaw
    """
    pitch, yaw, roll = pose_model.predict(face_img)
    return yaw
