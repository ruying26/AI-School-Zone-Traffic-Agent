"""Run the integrated agent. --scenario all runs five separate demonstrations."""
import argparse
import json
import time
from pathlib import Path
from controller import TrafficAgent
from environment import TrafficEnvironment, make_schedule, EW_GREEN
from memory import AgentMemory
from q_learning import QLearningAgent
from sensors import Camera
from settings import load_config, ROOT


def apply_demo_event(agent, step):
    # Deterministic faults make log evidence reproducible.
    if step == 28:
        agent.set_override(EW_GREEN)
    elif step == 32:
        agent.release_override()
    elif step == 35:
        agent.gateway.next_fault = 'timeout_before'
    elif step == 39:
        agent.gateway.next_fault = 'timeout_after'
    elif step == 43:
        agent.gateway.unreadable_polls = 3


def fmt_q(q_values):
    return ', '.join(f'{a}={v:.2f}' for a, v in q_values.items())


def print_details(event):
    """Extra explanation lines for the demo (only used with --verbose)."""
    for obs in event['observations']:
        reading = obs['reading']
        conf = reading.get('camera_confidence')
        conf_txt = f'{conf:.2f}' if isinstance(conf, (int, float)) else str(conf)
        score_txt = f"{obs['score']:.3f}" if obs['score'] is not None else 'n/a'
        valid_txt = 'valid' if obs['valid'] else 'INVALID/missing'
        age_txt = 'n/a' if obs.get('age_seconds') is None else f"{obs['age_seconds']:.0f}s"
        diff_txt = 'n/a' if obs.get('disagreement') is None else f"{obs['disagreement']:.2f}"
        print(f"    camera try {obs['attempt']}: confidence={conf_txt} ({valid_txt}), "
              f"age={age_txt}, disagreement={diff_txt}, "
              f"fuzzy reliability={score_txt} -> {obs['mode']}")
    if event.get('q_values'):
        print(f"    Q-values (allowed actions): {fmt_q(event['q_values'])}")
    if event.get('allowed_actions'):
        print(f"    safety-allowed actions: {', '.join(event['allowed_actions'])}")
    if event.get('proposed_action') != event.get('action'):
        print(f"    proposed={event['proposed_action']} but executed={event['action']} (safety rule)")
    print(f"    command: {event.get('command_status')} | reason: {event.get('reason')}")


def build_agent(config, model, output, seed=2026, scenario='all', untrained=False):
    if Path(model).exists():
        policy = QLearningAgent.load(model, config, seed)
    elif untrained:
        policy = QLearningAgent(config, seed)
    else:
        raise FileNotFoundError(f'Model not found: {model}. Run python train.py first, or use --untrained for fallback-only demonstration.')
    demand = scenario if scenario in ('rush', 'shift') else 'normal'
    env = TrafficEnvironment(config, make_schedule(seed, 36000, demand))
    camera = Camera(seed + 100000, 'uncertainty' if scenario == 'uncertainty' else 'normal')
    memory = AgentMemory(Path(output) / 'agent_memory.sqlite', config.to_dict(),
                         {'seed': seed, 'scenario': scenario, 'model': str(model),
                          'training_episodes': policy.training_episodes})
    return TrafficAgent(env, camera, policy, memory)


SCENARIOS = ('normal', 'rush', 'shift', 'uncertainty', 'failures')


def run_demo(config, args, scenario, output):
    """One self-contained run and its own evidence files."""
    agent = build_agent(config, args.model, output, args.seed, scenario, args.untrained)
    print(f'\n=== {scenario.upper()} scenario ===')
    print(f'Previous persisted runs: {agent.memory.previous_runs}; '
          f'trained episodes: {agent.policy.training_episodes}')
    last_mode = None
    try:
        for step in range(args.steps):
            if scenario == 'failures':
                apply_demo_event(agent, step)
            event = agent.step()
            state = event['state_after']
            important = bool(event['alert']) or event['mode'] != last_mode
            if step % args.every == 0 or important:
                marker = '>>' if event['mode'] != last_mode and last_mode is not None else '  '
                print(f"{marker}t={state['time']:4d} phase={state['phase']:16s} "
                      f"NS={state['ns_queue']:3d} EW={state['ew_queue']:3d} "
                      f"mode={event['mode']} action={event['action']}")
                if args.verbose:
                    print_details(event)
                if event['alert']:
                    print('  ALERT:', event['alert'])
            last_mode = event['mode']
            if args.delay:
                time.sleep(args.delay)
        metrics = agent.env.metrics()
        print('\n=== Run summary ===')
        for key, value in metrics.items():
            print(f'  {key}: {value:.3f}' if isinstance(value, float) else f'  {key}: {value}')
        output = Path(output)
        agent.memory.export(output / 'execution.jsonl')
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print('Evidence saved to', output)
        return metrics
    finally:
        agent.memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--model', default='artifacts/q_table.json')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--scenario', choices=('all',) + SCENARIOS, default='all')
    parser.add_argument('--untrained', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='print age, disagreement, fuzzy score, Q-values and decision reasons')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='seconds to pause after each step in a live demo')
    parser.add_argument('--every', type=int, default=1,
                        help='print every Nth step (alerts and mode changes always appear)')
    args = parser.parse_args()
    if not 1 <= args.steps <= 7200 or args.every < 1 or args.delay < 0:
        parser.error('--steps 1..7200, --every >= 1, --delay >= 0 required')
    try:
        config = load_config(args.config)
        if args.scenario == 'all':
            outcomes = {}
            for scenario in SCENARIOS:
                outcomes[scenario] = run_demo(config, args, scenario,
                                              Path(args.output) / 'demos' / scenario)
            print('\n=== All-scenario comparison (simulation only) ===')
            print('Scenario       Vehicle delay  Ped. wait  Max ped. wait  Over-target wait')
            for scenario, metrics in outcomes.items():
                print(f"{scenario:13s} {metrics['mean_vehicle_delay_so_far_s']:12.1f} "
                      f"{metrics['mean_pedestrian_wait_so_far_s']:10.1f} "
                      f"{metrics['max_pedestrian_wait_s']:14d} "
                      f"{metrics['pedestrian_wait_over_target_seconds']:16d}")
        else:
            run_demo(config, args, args.scenario, Path(args.output))
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
