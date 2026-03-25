# 🚦 IntelliAIDrive  
**Hybrid Autonomous Navigation: Visual Perception, NLP Command Intent, and RL Decision-Making**

IntelliAIDrive is a full-stack autonomous driving simulation platform engineered to recognize, classify, and respond to traffic signs through a multi-stage artificial intelligence pipeline. It bridges the gap between algorithmic decision-making and human oversight by combining Deep Learning vision pipelines with NLP-driven Reinforcement Learning.

---

## 🧠 System Architecture

Our architecture decouples visual feature extraction from the core navigation policy to optimize decision-making efficiency:

- **Visual Perception (YOLOv8n + ResNet18):**  
  YOLOv8n handles real-time detection and localization of traffic signs. These cropped bounding boxes are passed to a modified ResNet18 backbone (with an Identity layer) to extract dense, 512-dimensional feature embeddings.

- **Dynamic Decision Making (PPO RL Agent):**  
  A Proximal Policy Optimization (PPO) agent operating in a custom Gymnasium environment maps the 512-dimensional visual embeddings to optimal driving actions.

- **Human-in-the-Loop NLP:**  
  An intent classifier processes text-based user driving commands (e.g., "drive safely" vs. "drive aggressively") to dynamically adjust the RL agent's reward weights on the fly without violating absolute safety constraints.

---

## 🚀 Quick Start & Reproducibility

### 1. Full Pipeline Reproduction

To ensure academic rigor and strict reproducibility, the entire pipeline (environment setup, baseline training, and evaluation) can be executed via a single command:

```bash
make repro
```

---

### 2. Standalone Real-Time Detection (CLI)

To run the YOLOv8n + ResNet18 detection pipeline in real-time directly in a window:

```bash
python3 src/video_cli.py
```

---

### 3. Full-Stack Application (Backend + Frontend)

```bash
bash run.sh
```

---

## 📊 Documentation & Setup

For a deep dive into our methodology, data sourcing, and model evaluations, please review our formal documentation:

- **[Data Preparation & Preprocessing](docs/data_preprocessing.md)**  
  Detailed breakdown of our dataset validation, YOLO spatial filtering, and ResNet tensor transformations.

- **[Ethics Statement & Risk Register](docs/ethics_statement.md)** *(Coming Soon)*

- **[Model Card](docs/model_card.md)** *(Coming Soon)*

---

## 📁 Repository Structure

```text
IntelliAIDrive/
├── src/           # YOLO/ResNet Vision pipelines and API Backend
├── models/        # Saved model weights (.pt files)
├── rl/            # PPO Agent and Custom Gymnasium Environment
├── experiments/   # Jupyter notebooks for ablations and baselines
├── frontend/      # React + Tailwind Dashboard
└── docs/          # Project documentation, Ethics, and Data Prep
```

---

## 👥 The Team (6INTELSY)

- **Graciella Mhervie D. Jimenez** — Project Lead & Integration  
- **Jenica Sarah B. Tongol** — Data & Ethics Lead  
- **Arron Kian M. Parejas** — Modeling Lead (CNN/NLP/RL)  
- **Jian Kalel D. Marquez** — Evaluation & MLOps Lead  

---

## 📄 License

MIT License