"""Train through the integrated reliability-aware agent; persist model and evidence.

What this file does (in plain words):
  1. Creates an empty Q-learning "brain" (the Q-table).
  2. Runs many simulated school-zone episodes. In each one the agent controls the
     traffic light, gets a reward/penalty for each decision, and updates the Q-table.
  3. Saves the learned Q-table (q_table.json), a per-episode results table
     (training.csv) and the settings used (training_config.json).
Run it with:  python train.py        (quick test: python train.py --episodes 10 --steps 40)
"""
import argparse
from collections import Counter   # counts how often each control mode was used
import csv
import json
from pathlib import Path
from controller import TrafficAgent            # the agent: camera -> fuzzy -> Q-learning -> safety
from environment import TrafficEnvironment, make_schedule   # the simulated junction + traffic arrivals
from memory import AgentMemory                 # working/long-term memory of decisions
from q_learning import QLearningAgent          # the tabular Q-learning policy
from sensors import Camera                     # synthetic camera that adds noise/missing data
from settings import load_config, ROOT         # reads config.json


def train(config, output, episodes=None, steps=None):
    # Make sure the output folder (default: artifacts/) exists.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Use command-line values if given; otherwise use config.json (1500 episodes x 120 steps).
    episodes = episodes or config.training_episodes
    steps = steps or config.episode_steps
    if episodes <= 0 or steps <= 0:
        raise ValueError('Episodes and steps must be positive')

    # ONE Q-learning policy is created and shared by all episodes, so what it learns
    # in episode 1 is carried into episode 2, 3, ... (this is how it improves).
    policy = QLearningAgent(config, config.training_seed)
    rows = []   # one results row per episode -> becomes training.csv
    # The four situations are rotated so the agent practises all of them:
    # normal traffic, rush hour, demand shifting, and unreliable camera.
    scenarios = ('normal', 'rush', 'shift', 'uncertainty')

    for episode in range(episodes):
        # A different seed each episode gives different random traffic; results are reproducible.
        seed = config.training_seed + episode
        scenario = scenarios[episode % len(scenarios)]   # cycle through the 4 scenarios

        # Fresh simulated junction and traffic schedule for this episode (steps * 5 s long).
        env = TrafficEnvironment(config, make_schedule(seed, steps * config.step_seconds, scenario))
        memory = AgentMemory()   # no file given -> temporary memory, not saved to SQLite
        # Build the agent: same policy object every time, new environment and camera noise.
        agent = TrafficAgent(env, Camera(seed + 100000, scenario), policy, memory)

        # Epsilon = chance of trying a RANDOM action (exploration).
        # It starts at 1.0 (all random) and shrinks by 0.5% each episode down to 0.05,
        # so the agent explores early and exploits what it learned later.
        epsilon = max(config.epsilon_end, config.epsilon_start * config.epsilon_decay ** episode)

        total_reward = 0.0       # sum of rewards this episode (always negative: waiting is a cost)
        modes = Counter()        # how many steps were Q-learning vs fallback vs override, etc.

        for step in range(steps):
            # One decision cycle: observe -> fuzzy reliability -> choose action -> safety check
            # -> execute -> verify -> reward -> Q-table update (training=True enables learning).
            # terminal=True on the last step tells Q-learning there is no "next" state.
            event = agent.step(training=True, epsilon=epsilon, terminal=step == steps - 1)
            total_reward += event['reward']
            modes[event['mode']] += 1

        # Save this episode's summary. **env.metrics() adds waiting times, queues, switches,
        # safety_violations, etc. (these are the columns you see in Excel).
        rows.append({'episode': episode + 1, 'scenario': scenario, 'seed': seed,
                     'epsilon': epsilon, 'total_reward': total_reward, 'states': len(policy.table),
                     'learning_steps': modes['Q-learning permitted'], **env.metrics()})

        # Print progress every 100 episodes (and at the end).
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            print(f'Episode {episode + 1}/{episodes}: reward={total_reward:.1f}, states={len(policy.table)}')

    # ---- Save everything ----
    policy.training_episodes = episodes
    policy.save(output / 'q_table.json')      # the trained model that main.py loads at runtime

    # training.csv: the per-episode table you open in Excel (learning curve evidence).
    with (output / 'training.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # training_config.json: the exact settings and seeds used, so the run can be reproduced.
    (output / 'training_config.json').write_text(json.dumps(
        {'settings': config.to_dict(), 'episodes': episodes, 'steps': steps,
         'seed_start': config.training_seed, 'seed_end': config.training_seed + episodes - 1,
         'scenarios': scenarios}, indent=2), encoding='utf-8')
    return policy


if __name__ == '__main__':
    # Command-line options so you can change settings without editing the code.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))   # settings file
    parser.add_argument('--output', default='artifacts')                 # where to save results
    parser.add_argument('--episodes', type=int)                          # override number of episodes
    parser.add_argument('--steps', type=int)                             # override steps per episode
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.episodes, args.steps)
