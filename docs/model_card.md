# Model Card: IntelliAIDrive (v0.9)

## Model Description
IntelliAIDrive is a hybrid autonomous navigation agent that combines a CNN feature extractor (YOLOv8n) with a Reinforcement Learning policy (PPO). Its purpose is to accurately detect and classify traffic signs, dynamically adjusting driving behavior based on visual cues and simulated text commands.

## Model Details
- **Model Type:** Convolutional Neural Network (CNN) + Deep Reinforcement Learning (PPO)
- **Architecture:** YOLOv8n (Feature Extractor / Detection Backbone) feeding into a Gymnasium Discrete Action Space.
- **Pre-trained Weights:** YOLOv8n (COCO) fine-tuned on traffic data.

## Intended Use
- **Primary Use Cases:** Simulated traffic sign recognition, object detection, and RL-based decision-making.
- **Target Users:** Academic researchers evaluating hybrid vision/RL pipelines. (Strictly not for real-world vehicle control).

## Training Data
- **Data Sources:** Kaggle Traffic Sign Dataset & Car/Traffic Signs Detection Dataset
- **Preprocessing:** Bounding box normalization, image resizing, and feature embedding extraction.

## Performance
- **Evaluation Metrics:** Accuracy, F1, Precision, Recall
- **Results:** - RL Agent Accuracy: 94.1%
  - F1 Score: 91.9%
  - CNN Backbone Accuracy: ~99.28%
![Metrics](metrics.png)
![Confusion Matrix](conf-matrix.png)
