import os
import pickle
import numpy as np

class RLAgent:
    """
    Hybrid Reinforcement Learning Meta-Controller (AGGRESSIVE MODE)
    Controls:
    - ML confidence boost
    - Risk multiplier
    - Trade frequency
    """

    def __init__(self, policy_path="rl_policy.pkl", aggressive=True):
        self.policy_path = policy_path
        self.aggressive = aggressive

        # === RL STATE ===
        self.state = {
            "win_streak": 0,
            "loss_streak": 0,
            "risk_multiplier": 1.4 if aggressive else 1.0,
            "confidence_boost": 0.12 if aggressive else 0.05,
            "skip_trading": False
        }

        self.load()

    # ================= POLICY IO =================
    def save(self):
        with open(self.policy_path, "wb") as f:
            pickle.dump(self.state, f)

    def load(self):
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "rb") as f:
                self.state = pickle.load(f)

    # ================= RL ACTION =================
    def get_action(self):
        return self.state

    # ================= RL LEARNING =================
    def update_after_trade(self, pnl: float):
        """
        AGGRESSIVE reward logic
        """
        if pnl > 0:
            self.state["win_streak"] += 1
            self.state["loss_streak"] = 0

            # Scale up fast on wins
            self.state["risk_multiplier"] = min(
                2.0, self.state["risk_multiplier"] + 0.1
            )
            self.state["confidence_boost"] = min(
                0.25, self.state["confidence_boost"] + 0.02
            )

        else:
            self.state["loss_streak"] += 1
            self.state["win_streak"] = 0

            # Pull back on losses
            self.state["risk_multiplier"] = max(
                0.6, self.state["risk_multiplier"] - 0.15
            )
            self.state["confidence_boost"] = max(
                0.02, self.state["confidence_boost"] - 0.03
            )

            # Emergency brake after streak of losses
            if self.state["loss_streak"] >= 3:
                self.state["skip_trading"] = True

        # Auto re-enable trading after recovery
        if self.state["skip_trading"] and self.state["win_streak"] >= 2:
            self.state["skip_trading"] = False

        self.save()
