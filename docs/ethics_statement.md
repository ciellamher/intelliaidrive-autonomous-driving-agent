# Ethics Statement & Risk Register: IntelliAIDrive

## 1. Purpose & System Context
IntelliAIDrive is designed to demonstrate the potential of hybrid AI in autonomous driving, integrating visual perception (YOLOv8n + ResNet18), Natural Language Processing (NLP) for command intent, and a Proximal Policy Optimization (PPO) Reinforcement Learning agent. It is intended strictly for academic research and educational purposes.

## 2. Ethics Risk Register
The deployment of autonomous driving simulations introduces specific ethical and safety-critical challenges. The top three risks and our corresponding mitigations are:

### Risk 1: Safety-Critical Limits of Simulation
* **Description:** RL agents trained in simplified grid environments do not capture the unpredictable, high-stakes nature of real-world driving. There is a risk that simulation success is falsely equated with real-world readiness.
* **Mitigation:** We have implemented strict, clear disclaimers across the GitHub repository and Model Card stating that this system is a proof-of-concept simulation only. It is explicitly not validated, tested, or intended for real-world vehicle control or deployment.

### Risk 2: Environmental Bias and Domain Shift
* **Description:** The CNN components may learn to recognize traffic signs under ideal lighting but fail during edge cases (e.g., nighttime, heavy rain, or glare). This creates an equitable performance risk where the system behaves unsafely in certain geographic or weather conditions.
* **Mitigation:** We utilize highly diverse traffic sign datasets, which include variable lighting and weather conditions. We also conduct slice analysis on failure cases to document and expose any domain-shift vulnerabilities.

### Risk 3: Unsafe NLP Command Execution
* **Description:** The NLP intent classifier modifies the RL agent's reward function based on text commands (e.g., "drive fast"). There is a risk that a misclassified or maliciously crafted command could instruct the agent to ignore critical safety rules, such as stop signs.
* **Mitigation:** The RL reward function is hard-coded with absolute safety constraints. While NLP commands can adjust efficiency weights (e.g., speed), heavy penalties for traffic violations remain immutable, ensuring the agent cannot be "prompted" into dangerous behavior.

## 3. Data Ethics & Privacy
* **Provenance:** The models are trained on publicly available datasets from Kaggle:
    * [Traffic Sign Classification](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification)
    * [Traffic Signs Detection](https://www.kaggle.com/datasets/pkdarabi/cardetection)
* **Privacy:** Because these datasets feature street signs and public road artifacts rather than individuals, Personally Identifiable Information (PII) and human consent risks are effectively minimized. We confirm no PII is stored or processed.

## 4. Environmental Impact
* **Optimization:** Training workflows were optimized to minimize computational cycles and the associated carbon footprint.
* **Efficiency:** By decoupling the vision pipeline and utilizing lightweight models like YOLOv8n paired with a modified ResNet18 feature extractor, we ensure energy-efficient inference and significantly reduce computational overhead during deployment.    