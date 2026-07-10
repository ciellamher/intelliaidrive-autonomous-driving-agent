**intelliaidrive-autonomous-driving-agent**
A full-stack autonomous driving simulation platform combining Traffic Sign Recognition and Reinforcement Learning to navigate complex virtual environments.

### 01 — INSTALL

Bash
> git clone https://github.com/ciellamher/intelliaidrive-autonomous-driving-agent.git
> cd intelliaidrive-autonomous-driving-agent
> pip install -r requirements.txt

---

### 02 — USE

> python run_simulation.py --mode train
> python run_simulation.py --mode eval

Switch between training the agent and evaluating its driving performance.

---

### 03 — WHAT'S INSIDE

- `run_simulation.py` — The core environment controller and renderer.
- `agent.py` — Reinforcement learning models and state-action policies.
- `vision/` — Computer vision modules for detecting signs and objects.

---

### 04 — LICENSE

MIT
