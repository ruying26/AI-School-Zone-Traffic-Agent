"""Tabular Q-learning with legal-action masking and JSON persistence.

What this file does (in plain words):
  - Q-learning keeps a table: for every SITUATION (state) it stores a score (Q-value) for
    every ACTION. A higher score means "this action has worked out better in this situation".
  - encode_traffic_state() turns the junction's details into a small, simple "situation label".
  - QLearningAgent chooses actions, learns from rewards, and saves/loads the table as JSON.
  - "Legal-action masking": the agent only ever picks from the actions the safety rules allow,
    so the AI can never choose an unsafe signal change.
"""
import json
import math
from pathlib import Path
import random

# The four things the agent can ask the signal to do.
ACTIONS = ('KEEP', 'NS_GREEN', 'EW_GREEN', 'PEDESTRIAN_WALK')


def encode_traffic_state(state):
    # Compress the detailed junction state into a small label. Grouping similar situations
    # together keeps the Q-table small enough to learn from in 1,500 episodes.
    def queue_level(size):
        # Queue length in vehicles -> LOW (0-4), MEDIUM (5-14), HIGH (15+)
        return 'LOW' if size < 5 else 'MEDIUM' if size < 15 else 'HIGH'
    return (queue_level(state['ns_queue']),            # 1) North-South queue level
            queue_level(state['ew_queue']),            # 2) East-West queue level
            bool(state['pending_request']),            # 3) has a pedestrian pressed the button?
            min(state['ped_wait_s'] // 10, 6),         # 4) pedestrian wait in 10 s buckets (0 to 6)
            state['phase'],                            # 5) current signal phase (e.g. NS_GREEN)
            min(state['phase_elapsed_s'] // 5, 9))     # 6) time in this phase in 5 s buckets (0 to 9)


class QLearningAgent:
    def __init__(self, config, seed=0):
        self.config = config            # holds alpha, gamma (learning settings)
        self.table = {}                 # the Q-table: {situation: {action: score}}
        self.rng = random.Random(seed)  # seeded random generator -> reproducible choices
        self.training_episodes = 0      # how many episodes this model was trained for

    def values(self, state):
        # Look up the scores for a situation. If the situation was never seen, return
        # all zeros WITHOUT adding it to the table.
        # Read-only lookup: evaluation must not initialise or update Q-table entries.
        return self.table.get(tuple(state), dict.fromkeys(ACTIONS, 0.0))

    def choose(self, state, allowed, epsilon=0.0):
        # 'allowed' = the actions the safety rules permit right now (the "mask").
        if not allowed or any(a not in ACTIONS for a in allowed):
            raise ValueError('Need at least one valid allowed action')

        # Exploration: with probability epsilon, try a RANDOM allowed action.
        # (During training epsilon is high; at runtime it is 0, so this never happens.)
        if self.rng.random() < epsilon:
            return self.rng.choice(allowed)

        # Exploitation: pick the allowed action with the highest Q-value.
        # If several actions tie (e.g. an unseen state where all are 0), pick one randomly.
        values = self.values(state)
        best = max(values[a] for a in allowed)
        return self.rng.choice([a for a in allowed if values[a] == best])

    def update(self, state, action, reward, next_state, next_allowed, terminal=False):
        # The learning step, run after every training decision.
        if action not in ACTIONS:
            raise ValueError('Unknown action')

        # Get (or create) this situation's row in the table.
        values = self.table.setdefault(tuple(state), dict.fromkeys(ACTIONS, 0.0))

        # Best score the agent expects from the next situation, counting only allowed actions.
        # At the end of an episode (terminal) there is no next situation, so it is 0.
        future = 0.0 if terminal else max(self.values(next_state)[a] for a in next_allowed)

        # Q-learning formula:
        #   Q(s,a) = Q(s,a) + alpha * ( reward + gamma * best_future_Q - Q(s,a) )
        # i.e. move the old score a small step (alpha) toward "what I just got + what I expect later".
        values[action] += self.config.alpha * (reward + self.config.gamma * future - values[action])

    def save(self, path):
        # Write the learned table (plus the settings it was trained with) to a JSON file.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'training_episodes': self.training_episodes,
                   'config': self.config.to_dict(),
                   'entries': [{'state': list(s), 'values': v} for s, v in self.table.items()]}
        # Write to a temporary file first, then rename it. If the program crashes mid-save,
        # the old good q_table.json is not left half-written/corrupted.
        # allow_nan=False refuses to save broken numbers (NaN/infinity).
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(path)

    @classmethod
    def load(cls, path, config, seed=0):
        # Read a saved model back in. A classmethod: call it as QLearningAgent.load(...).
        payload = json.loads(Path(path).read_text(encoding='utf-8'))

        # Safety check: refuse a model trained with different settings (or file format).
        # This is why you MUST retrain after changing config.json.
        if payload.get('version') != 1 or payload.get('config') != config.to_dict():
            raise ValueError('Model version/config differs; retrain with this configuration')

        agent = cls(config, seed)
        for entry in payload['entries']:
            state, values = entry['state'], entry['values']
            # Validate every row: the situation has 6 parts, all 4 actions are present,
            # and every score is a real finite number.
            if len(state) != 6 or set(values) != set(ACTIONS) or not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Invalid Q-table entry')
            agent.table[tuple(state)] = values   # JSON lists -> tuples so they can be dictionary keys
        agent.training_episodes = payload['training_episodes']
        return agent
