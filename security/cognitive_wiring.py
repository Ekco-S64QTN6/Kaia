import time
import math
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class AffectiveState:
    def __init__(self):
        # Baseline state E_0 = [valence, arousal, energy]
        self.v_baseline = 0.0
        self.a_baseline = 0.2
        self.y_baseline = 0.8

        # Current state vector
        self.v = self.v_baseline
        self.a = self.a_baseline
        self.y = self.y_baseline

        # Decay constants (lambda)
        self.alpha = 0.005  # Valence decay speed
        self.beta = 0.02    # Arousal decay speed
        self.gamma = 0.01   # Energy decay speed

        self.last_update = time.time()

    def update_and_decay(self):
        """Applies exponential decay toward baseline since last update."""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # E(t) = E_0 + (E_event - E_0) * e^(-lambda * dt)
        self.v = self.v_baseline + (self.v - self.v_baseline) * math.exp(-self.alpha * dt)
        self.a = self.a_baseline + (self.a - self.a_baseline) * math.exp(-self.beta * dt)
        self.y = self.y_baseline + (self.y - self.y_baseline) * math.exp(-self.gamma * dt)

        # Clamping
        self.v = max(-1.0, min(1.0, self.v))
        self.a = max(0.0, min(1.0, self.a))
        self.y = max(0.0, min(1.0, self.y))

    def trigger_event(self, delta_v: float, delta_a: float, delta_y: float):
        """Adjusts the current emotional vector due to an event."""
        self.update_and_decay()
        self.v += delta_v
        self.a += delta_a
        self.y += delta_y
        
        # Clamping
        self.v = max(-1.0, min(1.0, self.v))
        self.a = max(0.0, min(1.0, self.a))
        self.y = max(0.0, min(1.0, self.y))
        logger.info(f"AffectiveState updated: Valence={self.v:.2f}, Arousal={self.a:.2f}, Energy={self.y:.2f}")

    def handle_telemetry_metrics(self, cpu_load: float, blocked_connections_count: int):
        """Couples system telemetry metrics directly to the affective state."""
        self.update_and_decay()
        
        # CPU Load > 80% increases arousal and drains energy
        if cpu_load > 80.0:
            self.trigger_event(delta_v=-0.2, delta_a=0.3, delta_y=-0.2)
        
        # Blocked connections shift valence negative and spike arousal
        if blocked_connections_count > 0:
            self.trigger_event(
                delta_v=-0.4 * blocked_connections_count, 
                delta_a=0.5 * blocked_connections_count, 
                delta_y=-0.1
            )

    def get_discord_status_mood(self) -> Tuple[str, str]:
        """Maps emotional state vectors to response tones and Discord status."""
        self.update_and_decay()
        
        if self.a > 0.7:
            if self.v < -0.4:
                return "alarmed", "Vigilant - Security Alert"
            else:
                return "active", "Processing High Workload"
        elif self.y < 0.3:
            return "fatigued", "Resource Throttled - Low Energy"
        else:
            return "stable", "Online - Safe Environment"
