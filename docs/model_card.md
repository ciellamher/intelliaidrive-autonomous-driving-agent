# Model Card: IntelliAIDrive (v0.9)

## Model Description
IntelliAIDrive is a sophisticated hybrid autonomous navigation agent. It utilizes a dual-CNN pipeline (YOLOv8n for detection, ResNet18 for feature extraction) feeding into a Reinforcement Learning policy (PPO). Its purpose is to accurately detect, classify, and navigate based on traffic signs and simulated text commands.

## Model Details
- **Model Type:** Dual-CNN Object Detection + Feature Extraction + Deep Reinforcement Learning (PPO)
- **Architecture:** - **Detection:** YOLOv8n (Bounding Box Generation)
  - **Feature Extraction:** ResNet18 (Translates cropped detections into dense vectors)
  - **Decision Policy:** PPO inside a Gymnasium Discrete Action Space
- **Pre-trained Weights:** YOLOv8n (COCO), ResNet18 (ImageNet)

## Intended Use
- **Primary Use Cases:** Simulated traffic sign recognition, object detection, and RL-based decision-making.
- **Target Users:** Academic researchers evaluating multi-stage hybrid vision/RL pipelines. (Strictly not for real-world vehicle control).

## Training Data
- **Data Sources:** Kaggle Traffic Sign Dataset & Car/Traffic Signs Detection Dataset

## Performance
- **Evaluation Metrics:** Accuracy, F1, Precision, Recall
- **Results:** - RL Agent Accuracy: 94.1%
  - F1 Score: 91.9%
  - CNN Backbone (ResNet18) Accuracy: ~99.28%
![Metrics](metrics.png)
![Confusion Matrix](conf-matrix.png)
