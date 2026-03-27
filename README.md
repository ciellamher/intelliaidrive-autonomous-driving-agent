# 🚦 IntelliAIDrive
**Traffic Sign Classification and Reinforcement Learning for Simulated Autonomous Navigation**

IntelliAIDrive is a simulation-based AI project that combines traffic sign classification and reinforcement learning for autonomous navigation research. The system uses a lightweight YOLOv8n classification model to recognize traffic signs, then integrates this structured information into a reinforcement learning environment for decision-making.

The project was developed as a final requirement for **6INTELSY AY 2025-2026**.

---

## Overview

The goal of IntelliAIDrive is to explore how structured visual recognition can support downstream navigation decisions more effectively than raw image input alone. Instead of forcing the decision-making stage to learn directly from pixels, the system first performs traffic sign classification and then passes the resulting information into a simulated RL setup.

This modular design makes the pipeline easier to interpret, lighter to train, and more suitable for controlled academic experimentation.

---

## Final Results

The final evaluated model achieved:

- **Accuracy:** 95.08%
- **F1-Score:** 93.38%
- **Precision:** 92.08%
- **Recall:** 95.08%

Additional outputs included:

- YOLO training curves
- Final metrics bar chart
- Confusion matrix
- Precision-recall curves for selected classes

---

## Project Components

- **Traffic Sign Classification:** YOLOv8n classification model trained on a multi-class traffic sign dataset
- **Reinforcement Learning:** PPO-based decision-making in a simulated navigation environment
- **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and class-wise precision-recall analysis
- **Documentation:** Proposal, checkpoint report, final report, ethics statement, and model card

---

## Quick Start

### Run the project | create your own environment

```bash
#Windows
python -m venv .venv

#Mac
python3 -m venv .venv

#Activate the environment on Windows
.venv\Scripts\activate

#Activate the environment on Mac
source .venv/bin/activate
```

### Install dependencies manually

```bash
pip install -r requirements.txt
```

# Run the app
```bash
python3 app/intellidrive.py
#or
python app/intellidrive.py
```

---

## Repository Files

- `README.md` - project overview and quick start
- `run.sh` - one-command reproduction script
- `requirements.txt` - environment dependencies
- `docs/model_card.md` - model details, intended use, and limitations
- `docs/ethics_statement.md` - risks, mitigations, and responsible-use notes
- `docs/IntelliAIDrive_Proposal.pdf` - proposal
- `docs/IntelliAIDrive_Checkpoint_Report.pdf` - checkpoint report
- `docs/IntelliAIDrive_Final_Report.pdf` - final report

---

## Documentation

- **Final Report:** `docs/IntelliAIDrive_Final_Report.pdf`
- **Proposal:** `docs/IntelliAIDrive_Proposal.pdf`
- **Checkpoint Report:** `docs/IntelliAIDrive_Checkpoint_Report.pdf`
- **Model Card:** `docs/model_card.md`
- **Ethics Statement:** `docs/ethics_statement.md`

---

## Team

- Graciella Mhervie D. Jimenez - Project Lead & Integration
- Jenica Sarah B. Tongol - Data & Ethics Lead
- Arron Kian M. Parejas - Modeling Lead
- Jian Kalel D. Marquez - Evaluation & MLOps Lead

---

## License

MIT License