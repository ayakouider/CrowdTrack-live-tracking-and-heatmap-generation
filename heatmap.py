import cv2
import numpy as np
import torch

model = torch.hub.load('yolov5', 'custom', path=r'C:\Users\ayakb\Desktop\heatmap generation\best5_windows_compatible.pt', source='local')

# Initialize video capture (for webcam)
cap = cv2.VideoCapture(0)  # Change to a video file path for video files

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# Paths for heatmap, overlay, and bounding box images
heatmap_path = r"C:\Users\ayakb\Desktop\heatmap generation\heat_lab1.jpg"
overlay_path = r"C:\Users\ayakb\Desktop\heatmap generation\overlay.jpg"
boundingbox_path = r"C:\Users\ayakb\Desktop\heatmap generation\boundingbox.jpg"



def detect_crowd_density_live(frame, model):
    results = model(frame)

    # Extract bounding boxes, confidences, and class names
    boxes = results.xywh[0].cpu().numpy()
    confidences = results.xywh[0][:, 4].cpu().numpy()
    class_ids = results.xywh[0][:, 5].cpu().numpy()

    person_detections = [(box, conf) for box, conf, cls_id in zip(boxes, confidences, class_ids) if int(cls_id) == 0]

    if not person_detections:
        return frame, None, 0  # No detections

    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for box, confidence in person_detections:
        x, y, w, h = map(int, box[:4])
        y_coord, x_coord = np.ogrid[-30:31, -30:31]
        g = np.exp(-(x_coord**2 + y_coord**2) / (2 * 15**2))

        y_min = max(0, y - 30)
        y_max = min(height, y + 31)
        x_min = max(0, x - 30)
        x_max = min(width, x + 31)

        g_crop = g[:y_max - y_min, :x_max - x_min]
        heatmap[y_min:y_max, x_min:x_max] += g_crop

    heatmap = np.exp(-heatmap / 5)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    return overlay, heatmap_color, len(person_detections)


def process_live_video(model,input=0,output_path=None):
    cap = cv2.VideoCapture(input)  # or replace 0 with a video file path
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        overlay, heatmap, count = detect_crowd_density_live(frame, model)

        if overlay is not None:
            cv2.imshow("Crowd Heatmap Overlay", overlay)
            if out:
                out.write(overlay)

            print(f"Detected People: {count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

process_live_video(model,input=r"C:\Users\ayakb\Desktop\heatmap generation\meeting2.mp4",output_path=r"C:\Users\ayakb\Desktop\heatmap generation\heatmap.mp4")