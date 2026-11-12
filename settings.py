"""Shared, validated simulation and training settings. No real-road certification.

What this file does (in plain words):
  - Defines every adjustable number (signal timings, learning rates, reward penalties)
    in ONE place, the Config class.
  - Reads those numbers from config.json when the program starts.
  - Checks the numbers are sensible, so a typo in config.json stops the program with a
    clear error instead of silently giving strange results.
"""
from dataclasses import asdict, dataclass   # dataclass = a class that mainly stores settings
import json
from pathlib import Path

# Folder that contains this file; used to find config.json no matter where you run Python from.
ROOT = Path(__file__).resolve().parent


# frozen=True makes the settings read-only after creation, so no part of the
# program can change them by accident while it is running.
@dataclass(frozen=True)
class Config:
    # ---- Timing (all in simulated seconds) ----
    step_seconds: int = 5                    # one agent decision every 5 s
    min_green_seconds: int = 15              # a green must last at least this long before switching
    max_green_seconds: int = 45              # a green may not last longer than this
    fallback_green_seconds: int = 25         # green time used by the safe fallback controller
    yellow_seconds: int = 3                  # yellow light duration
    all_red_seconds: int = 2                 # all-red gap between conflicting phases
    walk_seconds: int = 10                   # pedestrian WALK signal duration
    pedestrian_clearance_seconds: int = 10   # time for pedestrians to finish crossing
    max_pedestrian_wait_seconds: int = 30    # once a pedestrian has waited this long, WALK becomes the only allowed action
    service_interval_seconds: int = 2        # during a green, one vehicle leaves the queue every 2 s

    # ---- Q-learning (the AI's learning settings) ----
    alpha: float = 0.1            # learning rate: how strongly each new experience changes the Q-value
    gamma: float = 0.9            # discount factor: how much future rewards matter (0 = only now)
    epsilon_start: float = 1.0    # at the start, 100% random actions (explore)
    epsilon_end: float = 0.05     # at the end, only 5% random actions (mostly use what was learned)
    epsilon_decay: float = 0.995  # exploration shrinks by 0.5% each episode

    # ---- Reward penalties (what the agent is "punished" for) ----
    pedestrian_wait_weight: float = 3.0   # penalty per second of pedestrian waiting
    switch_penalty: float = 2.0           # penalty each time the signal changes phase

    # ---- Training size ----
    training_episodes: int = 1500   # number of simulated runs during training
    episode_steps: int = 120        # decisions per run (120 x 5 s = 600 s)
    training_seed: int = 42         # starting random seed, so results are reproducible

    def __post_init__(self):
        # Runs automatically right after the settings are loaded: this is the validation step.

        # 1) All time/count settings must be whole numbers greater than zero.
        integer_fields = ('step_seconds', 'min_green_seconds', 'max_green_seconds',
                          'fallback_green_seconds', 'yellow_seconds', 'all_red_seconds',
                          'walk_seconds', 'pedestrian_clearance_seconds',
                          'max_pedestrian_wait_seconds', 'service_interval_seconds',
                          'training_episodes', 'episode_steps')
        for name in integer_fields:
            value = getattr(self, name)
            # "type(value) is not int" also rejects decimals like 5.5 and True/False.
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        # 2) The fallback green time must sit between the minimum and maximum green.
        if not self.min_green_seconds <= self.fallback_green_seconds <= self.max_green_seconds:
            raise ValueError('Require min_green <= fallback_green <= max_green')

        # 3) Learning rate must be in (0, 1]; discount must be in [0, 1).
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid alpha or gamma')

        # 4) Exploration must go from a high value down to a lower one, both within 0..1.
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Invalid exploration limits')
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError('Invalid epsilon decay')

        # 5) Penalties cannot be negative (a negative penalty would act like a bonus).
        if self.pedestrian_wait_weight < 0 or self.switch_penalty < 0:
            raise ValueError('Reward penalties cannot be negative')

    def to_dict(self):
        # Turns the settings into a normal dictionary so they can be saved to JSON
        # (used by training_config.json and the SQLite run record).
        return asdict(self)


def load_config(path=None):
    # Read config.json (or another file given by --config) and build a validated Config.
    # Config(**...) fills each setting from the JSON; any name missing from the file keeps its
    # default above, and an unknown name or invalid value raises an error.
    with open(path or ROOT / 'config.json', encoding='utf-8') as stream:
        return Config(**json.load(stream))
