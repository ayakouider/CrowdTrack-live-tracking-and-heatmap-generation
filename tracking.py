from torchvision import transforms
from torch.nn import functional as F
import torch
import torchreid
from tqdm import tqdm
from trackers import DeepSORTTracker
import supervision as sv
from supervision import Detections
import cv2
import numpy as np

class DeepSORTTorchreidExtractor:
    def __init__(self, model_name='osnet_ain_x1_0', device='cpu'):
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        ).to(device)
        self.model.eval()

        # Same transform as used in torchreid
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __call__(self, image_crops):
        with torch.no_grad():
            batch = torch.stack([self.transform(img) for img in image_crops]).to(self.device)
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy()

    def extract_features(self, frame, detections):
        """
        Extract features for bounding boxes in `detections.xyxy` from `frame`.
        """
        crops = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = self.transform(crop)
            crops.append(crop)

        if not crops:
            return []

        import torch
        batch = torch.stack(crops).to(self.device)

        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

extractor = DeepSORTTorchreidExtractor(
    model_name='osnet_x1_0',
    device='cpu'
)

box_annotator = sv.BoxAnnotator(
    color= sv.Color.GREEN,
    color_lookup=sv.ColorLookup.TRACK)


label_annotator = sv.LabelAnnotator(
    color=sv.Color.WHITE,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_scale=0.4)

extractor.normalize=True

tracker = DeepSORTTracker(feature_extractor=extractor)
tracker.max_age = 15  # Reduce max_age to prevent false-positive tracking
tracker.n_init = 3 # Increase n_init for more robust object tracking
tracker.max_iou_distance = 0.5  # Lower max_iou_distance for stricter matching

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

model = torch.hub.load('yolov5', 'custom', path=r'C:\Users\ayakb\Desktop\heatmap generation\best5_windows_compatible.pt', source='local')

def process_frame(frame):
    results = model(frame)

    # Get predictions as DataFrame (xywh format)
    detections_df = results.pandas().xywh[0]

    boxes = []
    confidences = []

    for _, row in detections_df.iterrows():
        if row['confidence'] < CONFIDENCE_THRESHOLD:
            continue
        if int(row['class']) != 0:  # Class 0 = person
            continue

        cx, cy, w, h = row['xcenter'], row['ycenter'], row['width'], row['height']

        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append([x1, y1, x2, y2])
        confidences.append(row['confidence'])

    # Create Detections object
    if len(boxes) == 0:
        detections = Detections.empty()
    else:
        detections = Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.zeros(len(boxes), dtype=int)
        ).with_nms(threshold=NMS_THRESHOLD)

        features = extractor.extract_features(frame, detections)
        detections.features = features
    # Update tracker
    tracks = tracker.update(detections=detections, frame=frame)
    return tracks

def track_people(model, device, input_path=0, output_path=None, show=False, save=True):
    cap = cv2.VideoCapture(input_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set default output path if not provided
    if output_path is None and save:
        output_path = "tracking_output.mp4"

    # Initialize video writer if saving is enabled
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracks = process_frame(frame)

        # Annotate the frame
        annotated_image = frame.copy()
        annotated_image = box_annotator.annotate(annotated_image, tracks)
        labels = [f"ID {id}" for id in tracks.tracker_id]
        annotated_image = label_annotator.annotate(annotated_image, tracks, labels=labels)

        # Save annotated frame to output video
        if save:
            out.write(annotated_image)

        # Display frame if enabled
        if show:
            cv2.imshow("Tracking", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    if save:
        out.release()
        print(f"Tracking output saved to: {output_path}")
    if show:
        cv2.destroyAllWindows()

device = torch.device("cpu")

track_people(model, device=device,input_path=r"C:\Users\ayakb\Desktop\heatmap generation\meeting1.mp4",output_path=r"C:\Users\ayakb\Desktop\heatmap generation\fifth.mp4", show=True, save=True)