"""Paired held-out evaluation with a frozen Q-table and identical arrival schedules."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from controller import TrafficAgent
from environment import TrafficEnvironment, make_schedule
from main import apply_demo_event
from memory import AgentMemory
from q_learning import QLearningAgent
from sensors import Camera
from settings import load_config, ROOT


def evaluate(config, model, output, seeds=10, steps=120, seed_start=50000):
    if seeds <= 0 or steps <= 0:
        raise ValueError('Seeds and steps must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in ('normal', 'rush', 'shift', 'uncertainty', 'failures'):
        for seed in range(seed_start, seed_start + seeds):
            schedule = make_schedule(seed, steps * config.step_seconds, scenario)
            for baseline in (True, False):
                policy = QLearningAgent.load(model, config, seed)
                frozen = json.dumps([{'state': s, 'v': v} for s, v in policy.table.items()], sort_keys=True)
                env = TrafficEnvironment(config, schedule)
                agent = TrafficAgent(env, Camera(seed + 100000, scenario), policy, AgentMemory(), baseline)
                modes, alerts = Counter(), 0
                for step in range(steps):
                    if scenario == 'failures':
                        apply_demo_event(agent, step)
                    event = agent.step()
                    modes[event['mode']] += 1
                    alerts += bool(event['alert'])
                assert frozen == json.dumps([{'state': s, 'v': v} for s, v in policy.table.items()], sort_keys=True), 'Evaluation changed Q-table'
                rows.append({'scenario': scenario, 'seed': seed,
                             'controller': 'fixed_time' if baseline else 'ai_agent',
                             'policy_steps': modes['Q-learning permitted'],
                             'fallback_steps': sum(n for m, n in modes.items() if 'fallback' in m.lower()),
                             'alerts': alerts, **env.metrics()})
    with (output / 'evaluation.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for scenario in ('normal', 'rush', 'shift', 'uncertainty', 'failures'):
        for controller in ('fixed_time', 'ai_agent'):
            subset = [r for r in rows if r['scenario'] == scenario and r['controller'] == controller]
            summary.append({'scenario': scenario, 'controller': controller,
                            **{k: statistics.mean(r[k] for r in subset) for k in (
                                'mean_vehicle_delay_so_far_s', 'mean_pedestrian_wait_so_far_s',
                                'max_pedestrian_wait_s', 'pedestrian_wait_over_target_seconds',
                                'pedestrians_started_over_target', 'pedestrians_currently_over_target',
                                'vehicles_remaining', 'safety_violations',
                                'policy_steps', 'fallback_steps', 'alerts')}})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'evaluation_config.json').write_text(json.dumps(
        {'settings': config.to_dict(), 'seeds': seeds, 'seed_start': seed_start, 'steps': steps,
         'model': str(model), 'learning': False, 'epsilon': 0.0}, indent=2), encoding='utf-8')
    for row in summary:
        print(f"{row['scenario']:12s} {row['controller']:10s} vehicle delay={row['mean_vehicle_delay_so_far_s']:.2f}s "
              f"pedestrian wait={row['mean_pedestrian_wait_so_far_s']:.2f}s "
              f"over-target seconds={row['pedestrian_wait_over_target_seconds']:.1f}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--model', default='artifacts/q_table.json')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed-start', type=int, default=50000)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.model, args.output, args.seeds, args.steps, args.seed_start)
