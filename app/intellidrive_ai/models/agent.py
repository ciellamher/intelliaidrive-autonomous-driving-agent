import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional

class PolicyNetwork(nn.Module):
    """
    Simple MLP for RL Decision Making.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_actions: int = 3):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class RLAgent:
    """
    RLAgent responsible for taking vision inputs and outputting driving intents.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.input_dim = 4 # [distance_to_stop, light_state_encoded, nearest_vehicle_dist, current_speed]
        self.num_actions = 4 # [STOP, SLOW, PROCEED, CAUTION]
        self.actions = ["STOP", "SLOW", "PROCEED", "CAUTION"]
        
        self.policy = PolicyNetwork(self.input_dim, 16, self.num_actions).to(self.device)
        self.policy.eval()
        
        if model_path:
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                print(f"[Agent] Loaded weights from {model_path}")
            except:
                print(f"[Agent] Warning: No model weights found at {model_path}, using randomized policy.")

    def _get_state(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Encodes semantic observations into a flat tensor for the model.
        """
        # Encode light state
        light_map = {"RED": 1.0, "YELLOW": 0.5, "GREEN": 0.0, "UNKNOWN": 0.0}
        light_val = light_map.get(observations.get("light_status", "UNKNOWN"), 0.0)
        
        # Normalize distances (0 to 1, where 1 is close, 0 is far)
        # Max distance assumed 1000px
        stop_dist = np.clip(1.0 - (observations.get("stop_sign_dist", 1000.0) / 1000.0), 0, 1.0)
        veh_dist = np.clip(1.0 - (observations.get("nearest_vehicle_dist", 1000.0) / 1000.0), 0, 1.0)
        speed = np.clip(observations.get("speed", 0.0) / 100.0, 0, 1.0)
        
        state = [stop_dist, light_val, veh_dist, speed]
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def decide(self, observations: Dict[str, Any]) -> Tuple[str, float]:
        """
        Heuristic-driven decision engine (Safe-by-design).
        1. STOP: Red Light or Stop Sign < 150px
        2. SLOW: Yellow Light or Vehicle/Person < 300px
        3. PROCEED: No Hazards, Green Light, or Unknown (GO by default)
        """
        light = observations.get("light_status", "UNKNOWN")
        stop_dist = observations.get("stop_sign_dist", 1000.0)
        veh_dist = observations.get("nearest_vehicle_dist", 1000.0)
        
        # 1. Critical Stop
        if light == "RED" or stop_dist < 150:
            return "STOP", 1.0
            
        # 2. Cautionary Slow
        if light == "YELLOW" or veh_dist < 300:
            return "SLOW", 0.9
            
        # 3. Proceed (GO) - Default state
        # In the absence of hazards or UNKNOWN light, we GO.
        return "GO", 1.0

if __name__ == "__main__":
    agent = RLAgent()
    obs = {"stop_sign_dist": 1000, "light_status": "GREEN", "nearest_vehicle_dist": 500, "speed": 60}
    action, conf = agent.decide(obs)
    print(f"Decided: {action} with confidence {conf:.2f}")
