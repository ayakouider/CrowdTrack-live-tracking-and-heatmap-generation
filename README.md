## AI-Based Crowd Monitoring Using YOLOv5 + DeepSORT
This repository contains the final AI implementation of our capstone project CrowdTrack:a real-time room occupancy and heatmap system using computer vision. 
The project was built with YOLOv5 for person detection, DeepSORT for tracking, and custom heatmap generation based on tracked movements.

### 📁 Repository Contents:
- yolov5_training.ipynb : Final YOLOv5 training notebook on fisheye-adapted dataset.
- tracking.py : live Tracking using Torchreid and DeepSORT.
- heatmap.py : live heatmap generation.
- Final presentation.ppt : project presentation.
- CapstoneFinal.docs: project report.

###  Project Overview:
This capstone project focuses on detecting, tracking, and visualizing the presence of individuals in indoor environments using fisheye camera footage.
The system generates real-time occupancy heatmaps and assigns unique IDs to people across frames for robust room monitoring.

# Main Libraries:
- YOLOv5
- DeepSORT
- Torch
- OpenCV
- matplotlib






