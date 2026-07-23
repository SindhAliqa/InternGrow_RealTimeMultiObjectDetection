from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Counting line position
line_y = 300

# Variables
count = 0
crossed_ids = set()
previous_positions = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track objects
    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()

    # Draw counting line
    cv2.line(annotated_frame, (0, line_y), (640, line_y), (0, 255, 0), 3)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):

            # Count only persons (COCO class 0)
            # Count only these objects
            
            if cls not in [39, 56, 63, 67]:
                continue

            x1, y1, x2, y2 = box

            center_y = int((y1 + y2) / 2)

            if track_id in previous_positions:

                prev_y = previous_positions[track_id]

                if prev_y < line_y <= center_y:

                    if track_id not in crossed_ids:
                        count += 1
                        crossed_ids.add(track_id)

            previous_positions[track_id] = center_y

    # Display count
    cv2.putText(
        annotated_frame,
        f"Person Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("Real-Time Object Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()