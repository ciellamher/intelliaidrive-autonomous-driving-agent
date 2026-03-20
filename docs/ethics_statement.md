# Ethics Statement: IntelliDrive AI (v0.9)

## Purpose
IntelliDrive AI is designed to demonstrate the potential of multi-stage computer vision and reinforcement learning in autonomous driving. It is intended for research and educational purposes only.

## Safety & Responsibility
* **Hybrid Decision Logic**: The system utilizes a multi-layered approach. YOLOv8n handles initial detection, while ResNet18 provides robust feature embeddings (achieving **~99.28% test accuracy**). Finally, the Reinforcement Learning agent (PPO) manages classification within a `Gymnasium`-based environment.
* **Simulation Sovereignty**: All training and evaluation occur within a controlled simulation. Real-world deployment is strictly prohibited as the RL reward shaping (currently optimized at a **+5/-3 ratio**) is tuned for simulated convergence, not human-safety critical environments.
* **Human Oversight**: We advocate for a "human-in-the-loop" approach where human operators maintain ultimate control and can intervene to override autonomous decisions. 

## Data Ethics & Model Integrity
* **Provenance**: The models are trained on publicly available datasets from Kaggle.
* **Accuracy & Transparency**: We acknowledge the performance gap between the CNN feature pipeline (~99.28%) and the RL agent (94.1%). We are committed to transparency regarding these margins, the current F1 Score of **91.9%**, and the potential for bias in detection across low-light or adverse weather conditions.
* **Privacy**: We confirm that no Personally Identifiable Information (PII) of real-world drivers, pedestrians, or vehicle owners is stored or processed within this project.

## Environmental Impact
* **Optimization**: Training workflows were optimized to minimize computational cycles. By utilizing pre-trained networks (YOLOv8n and ResNet18) to handle vision tasks before RL training, we significantly reduced the agent's exploration time and total carbon footprint.
