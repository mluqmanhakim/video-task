from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


def crop_with_padding(frame, box, padding=0.05):
    """
    Crop a bounding box while adding proportional padding.

    box: (x1, y1, x2, y2)
    padding: fraction of bounding-box width/height
    """
    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_width, x2 + pad_x)
    y2 = min(frame_height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def extract_person_crops(
    video_path,
    output_dir,
    model_path="yolo11n.pt",
    confidence_threshold=0.5,
    save_every_n_frames=15,
    max_images_per_person=30,
    min_crop_width=50,
    min_crop_height=100,
    padding=0.05,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_number = 0
    total_saved = 0

    # Number of images already saved for each tracking ID
    saved_per_track = {}

    while True:
        success, frame = video.read()

        if not success:
            break

        # Process only selected frames to reduce duplicate images
        if frame_number % save_every_n_frames != 0:
            frame_number += 1
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],  # COCO class 0 = person
            conf=confidence_threshold,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            frame_number += 1
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences,
        ):
            current_count = saved_per_track.get(track_id, 0)

            if current_count >= max_images_per_person:
                continue

            x1, y1, x2, y2 = box
            crop_width = x2 - x1
            crop_height = y2 - y1

            # Skip tiny detections that likely lack apron detail
            if crop_width < min_crop_width or crop_height < min_crop_height:
                continue

            person_crop = crop_with_padding(
                frame,
                box,
                padding=padding,
            )

            if person_crop is None or person_crop.size == 0:
                continue

            filename = (
                f"person_{track_id:05d}_"
                f"frame_{frame_number:08d}_"
                f"conf_{confidence:.2f}.jpg"
            )

            output_path = output_dir / filename

            saved = cv2.imwrite(
                str(output_path),
                person_crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if saved:
                saved_per_track[track_id] = current_count + 1
                total_saved += 1

        if frame_number % 300 == 0:
            print(
                f"Frame {frame_number}/{total_frames} | "
                f"People tracked: {len(saved_per_track)} | "
                f"Images saved: {total_saved}"
            )

        frame_number += 1

    video.release()

    print("\nExtraction completed")
    print(f"Video FPS: {fps:.2f}")
    print(f"Frames processed: {frame_number}")
    print(f"Tracking IDs found: {len(saved_per_track)}")
    print(f"Images saved: {total_saved}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tracked person crops from a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video",
    )
    parser.add_argument(
        "--output",
        default="dataset/unlabeled",
        help="Directory for cropped images",
    )
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help="Ultralytics detection model",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum person detection confidence",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Save detections once every N frames",
    )
    parser.add_argument(
        "--max-per-person",
        type=int,
        default=30,
        help="Maximum images saved for each tracking ID",
    )

    args = parser.parse_args()

    extract_person_crops(
        video_path=args.video,
        output_dir=args.output,
        model_path=args.model,
        confidence_threshold=args.confidence,
        save_every_n_frames=args.interval,
        max_images_per_person=args.max_per_person,
    )