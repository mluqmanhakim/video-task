import cv2
import numpy as np

shelf_a = np.array([[725, 340], [815, 240], [1060, 400], [1010, 500]], dtype=np.int32)
shelf_ab = np.array([[540, 435], [680, 355], [970, 550], [830, 720]], dtype=np.int32)
shelf_b = np.array([[215, 580], [442, 530], [640, 720], [285, 720]], dtype=np.int32)
shelf_c = np.array([[343, 460], [670, 315], [740, 360], [410, 530]], dtype=np.int32)
shelf_d = np.array([[148, 430], [260, 400], [580, 720], [285, 720]], dtype=np.int32)
all_shelfs = [shelf_a, shelf_ab, shelf_b, shelf_c, shelf_d]


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


def check_point_in_polygon(point, polygon):
    """
    Check whether a point is inside a polygon.
    """
    x, y = point
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def check_position_inside_any_shelf(position):
    for i, shelf in enumerate(all_shelfs):
        in_shelf = check_point_in_polygon(position, shelf)
        if in_shelf:
            return True
    return False


def get_shelf_interest(position, person_img, face_model, pose_model):
    face_crop = detect_face(face_model=face_model, person_img=person_img)
    yaw = None
    if face_crop is not None and face_crop.size > 0:
        yaw = detect_head_pose(pose_model=pose_model, face_img=face_crop)
    else:
        return None

    in_shelf_d = check_point_in_polygon(position, shelf_d)
    if in_shelf_d:
        if yaw >= 15 and yaw < 50:
            return "D"

    in_shelf_b = check_point_in_polygon(position, shelf_b)
    if in_shelf_b:
        if yaw > -40 and yaw < 15:
            return "B"

    in_shelf_ab = check_point_in_polygon(position, shelf_ab)
    if in_shelf_ab:
        if yaw >= 20 and yaw < 50:
            return "B"
        elif yaw > -20 and yaw < 25:
            return "A"

    in_shelf_c = check_point_in_polygon(position, shelf_c)
    if in_shelf_c:
        if yaw >= -30 and yaw <= 2:
            return "C"

    return None


def put_person_label(person_id, label, frame, label_position):
    label_text = f"ID {person_id}: {label}"
    x1, y1 = label_position
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
