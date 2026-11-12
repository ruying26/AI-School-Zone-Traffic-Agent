"""Automated tests for the school-zone agent.

What this file does (in plain words):
  Each test builds a tiny simulated junction, makes something happen (a bad camera, a
  failed command, a pedestrian waiting...) and checks the agent behaves safely.
  If all tests pass, the safety rules and the uncertainty handling are working.
Run with:  python -m unittest discover -s tests -v
"""

from pathlib import Path
import tempfile
import unittest

from controller import TrafficAgent
from environment import TrafficEnvironment, make_schedule, NS_GREEN, EW_GREEN, PEDESTRIAN_WALK
from fuzzy_logic import (calculate_fuzzy_reliability, select_control_mode,
                         observation_disagreement)
from memory import AgentMemory
from q_learning import QLearningAgent, encode_traffic_state
from sensors import Camera, valid_observation
from settings import Config


# A fake camera for testing. You give it a list of confidence values and it returns them
# one by one (None = camera offline). 'calls' counts how many times the agent looked.
class FixedCamera:
    def __init__(self, readings):
        self.readings = readings
        self.calls = 0

    def read(self, env, attempt=0):
        value = self.readings[min(self.calls, len(self.readings) - 1)]
        self.calls += 1
        reading = {'camera_available': value is not None, 'camera_confidence': value,
                   'timestamp': env.time, 'ns_queue': len(env.ns), 'ew_queue': len(env.ew),
                   'pedestrian_count': len(env.pedestrians)}
        return reading


# All tests live in this class. Every test starts with a fresh setUp() below.
class AgentTests(unittest.TestCase):
    # Runs before EVERY test: default settings, an empty junction (no traffic arrivals),
    # a blank Q-learning policy and a working-memory-only memory.
    def setUp(self):
        self.c = Config()
        self.env = TrafficEnvironment(self.c, [(0, 0, 0)] * 2000)
        self.policy = QLearningAgent(self.c, 1)
        self.memory = AgentMemory()

    # Helper: build an agent whose camera returns the given confidence values.
    def agent(self, readings=(0.9,)):
        return TrafficAgent(self.env, FixedCamera(readings), self.policy, self.memory)

    # Helper: put a fake Q-table entry for the current situation so that 'action'
    # has the highest score (10.0). Lets a test control what the AI 'prefers'.
    def teach(self, action):
        s = encode_traffic_state(self.env.state())
        self.policy.table[s] = dict.fromkeys(('KEEP', NS_GREEN, EW_GREEN, PEDESTRIAN_WALK), 0.0)
        self.policy.table[s][action] = 10.0

    # Good camera (0.9) -> Q-learning is allowed and the agent really executes the learned action.
    def test_reliable_camera_uses_learned_action(self):
        self.env.elapsed = 20
        self.teach(EW_GREEN)
        event = self.agent().step()
        self.assertEqual(event['mode'], 'Q-learning permitted')
        self.assertEqual(event['action'], EW_GREEN)
        self.assertEqual(event['state_after']['phase'], EW_GREEN)

    # First reading 0.6 (medium) -> agent looks again; second reading 0.9 -> Q-learning allowed.
    # Proves 're-observation' works (camera called twice).
    def test_medium_confidence_recovers_on_second_reading(self):
        self.teach('KEEP')
        agent = self.agent((0.6, 0.9))
        event = agent.step()
        self.assertEqual(agent.camera.calls, 2)
        self.assertEqual(event['mode'], 'Q-learning permitted')

    # Camera stays at 0.6 -> agent looks 3 times in total (1 + 2 retries), then gives up
    # and uses the safe fallback. Proves retries are limited, not endless.
    def test_uncertainty_retries_are_bounded_and_execute_fallback(self):
        self.env.elapsed = 25
        agent = self.agent((0.6,))
        event = agent.step()
        self.assertEqual(agent.camera.calls, 3)
        self.assertEqual(event['mode'], 'Safe fallback')
        self.assertEqual(event['action'], EW_GREEN)
        self.assertEqual(self.env.phase, EW_GREEN)

    # Camera offline (None) -> no retries, straight to fallback, no fuzzy score,
    # and nothing is stored as a 'last valid observation'.
    def test_missing_camera_fallback_without_retries(self):
        agent = self.agent((None,))
        event = agent.step()
        self.assertEqual(agent.camera.calls, 1)
        self.assertEqual(event['mode'], 'Safe fallback')
        self.assertIsNone(event['score'])
        self.assertIsNone(self.memory.last_observation)

    # Impossible confidence values (NaN, below 0, above 1) are rejected -> fallback.
    def test_bad_confidence_rejected(self):
        for bad in (float('nan'), -0.1, 1.1):
            event = self.agent((bad,)).step()
            self.assertEqual(event['mode'], 'Safe fallback')

    # A reading that is too old (more than 10 s) or from the future is rejected.
    def test_stale_or_future_reading_rejected(self):
        reading = Camera().read(self.env)
        self.assertFalse(valid_observation(reading, 11))
        reading['timestamp'] = 1
        self.assertFalse(valid_observation(reading, 0))

    # A situation never seen in training -> fallback (not a guess), and the Q-table is
    # left unchanged (inference never edits the learned model).
    def test_unknown_policy_state_uses_fallback_without_mutating_table(self):
        event = self.agent().step()
        self.assertEqual(event['mode'], 'Safe fallback: unseen state')
        self.assertEqual(self.policy.table, {})

    # Cannot change the light before the minimum green time has passed.
    def test_minimum_green_rejects_switch(self):
        self.assertFalse(self.env.command(EW_GREEN))
        self.assertEqual(self.env.phase, NS_GREEN)

    # A switch goes NS_GREEN -> YELLOW (3 s) -> ALL_RED (2 s) -> EW_GREEN, never directly,
    # and vehicles do not move during yellow/all-red.
    def test_explicit_yellow_and_all_red(self):
        self.env.elapsed = 15
        self.env.ns.append(0)
        self.assertTrue(self.env.command(EW_GREEN))
        self.assertEqual(self.env.phase, 'YELLOW')
        for _ in range(3):
            self.env.tick()
        self.assertEqual(self.env.phase, 'ALL_RED')
        self.assertEqual(len(self.env.ns), 1)
        self.env.tick()
        self.assertEqual(self.env.phase, 'ALL_RED')
        self.env.tick()
        self.assertEqual(self.env.phase, EW_GREEN)

    # Pedestrian sequence: WALK (10 s) -> PED_CLEARANCE (10 s) -> ALL_RED -> green.
    # No cars move and no vehicle phase can start while people are crossing.
    def test_walk_and_clearance_block_vehicles(self):
        self.env.elapsed = 15
        self.env.pedestrians.append(0)
        self.env.ns.append(0)
        self.assertTrue(self.env.command(PEDESTRIAN_WALK))
        for _ in range(5):
            self.env.tick()
        self.assertEqual(self.env.phase, PEDESTRIAN_WALK)
        self.assertFalse(self.env.command(NS_GREEN))
        for _ in range(10):
            self.env.tick()
        self.assertEqual(self.env.phase, 'PED_CLEARANCE')
        self.assertEqual(self.env.pedestrians_served, 1)
        self.assertEqual(len(self.env.ns), 1)
        for _ in range(10):
            self.env.tick()
        self.assertEqual(self.env.phase, 'ALL_RED')
        for _ in range(2):
            self.env.tick()
        self.assertEqual(self.env.phase, EW_GREEN)

    # If another pedestrian presses the button during WALK, that request is kept for next time.
    def test_new_request_during_walk_is_not_lost(self):
        self.env.elapsed = 15
        self.env.pedestrians.append(0)
        self.env.command(PEDESTRIAN_WALK)
        for _ in range(5):
            self.env.tick()
        self.env.pedestrians.append(self.env.time)
        for _ in range(10):
            self.env.tick()
        self.assertEqual(len(self.env.pedestrians), 1)
        self.assertEqual(self.env.pedestrians_served, 1)

    # An overdue pedestrian request forces the next safe available WALK action.
    def test_pedestrian_deadline_masks_keep(self):
        self.env.time = 60
        self.env.elapsed = 20
        self.env.pedestrians.append(0)
        self.assertEqual(self.env.safe_actions(), (PEDESTRIAN_WALK,))

    # After the maximum green time (45 s), KEEP is no longer allowed (must change).
    def test_max_green_masks_keep(self):
        self.env.elapsed = 45
        self.assertNotIn('KEEP', self.env.safe_actions())

    # A human override still cannot break safety rules (EW_GREEN is refused -> KEEP);
    # the camera is not used and nothing is learned during manual control.
    def test_manual_override_obeys_safety(self):
        agent = self.agent()
        agent.set_override(EW_GREEN)
        event = agent.step(training=True)
        self.assertEqual(event['mode'], 'Manual override')
        self.assertEqual(event['action'], 'KEEP')
        self.assertFalse(event['learning_updated'])
        self.assertEqual(agent.camera.calls, 0)

    # After a human releases control, the agent needs fresh confirmed controller state;
    # if readback fails it sends nothing, then runs one 'Resume verification' step first.
    def test_manual_release_requires_confirmation(self):
        agent = self.agent()
        agent.set_override('KEEP')
        agent.release_override()
        agent.gateway.unreadable_polls = 1
        event = agent.step()
        self.assertTrue(agent.resume_pending)
        self.assertEqual(event['command_status'], 'not_sent')
        event = agent.step()
        self.assertEqual(event['mode'], 'Resume verification')
        self.assertFalse(agent.resume_pending)

    # Command times out BEFORE executing: phase unchanged, only ONE command sent (no blind
    # retry), and this step is not used for learning.
    def test_timeout_before_command_does_not_change_phase_or_retry(self):
        self.env.elapsed = 20
        self.teach(EW_GREEN)
        agent = self.agent()
        agent.gateway.next_fault = 'timeout_before'
        event = agent.step(training=True)
        self.assertEqual(agent.gateway.command_count, 1)
        self.assertEqual(self.env.phase, NS_GREEN)
        self.assertEqual(event['command_status'], 'timeout')
        self.assertFalse(event['learning_updated'])

    # Command executes but the acknowledgement is lost: the agent reads back the REAL state
    # (YELLOW), does not resend, and excludes the step from learning.
    def test_lost_ack_after_execution_uses_actual_state(self):
        self.env.elapsed = 20
        self.teach(EW_GREEN)
        agent = self.agent()
        agent.gateway.next_fault = 'timeout_after'
        event = agent.step(training=True)
        self.assertEqual(agent.gateway.command_count, 1)
        self.assertEqual(event['confirmed_state']['phase'], 'YELLOW')
        self.assertEqual(self.env.phase, EW_GREEN)
        self.assertFalse(event['learning_updated'])

    # While the controller state cannot be read, NO remote command is sent; once readable
    # again, commands resume.
    def test_unreadable_controller_blocks_remote_commands_then_recovers(self):
        agent = self.agent()
        agent.gateway.unreadable_polls = 2
        for _ in range(2):
            event = agent.step()
            self.assertEqual(event['command_status'], 'not_sent')
        self.assertEqual(agent.gateway.command_count, 0)
        agent.step()
        self.assertEqual(agent.gateway.command_count, 1)

    # Readback fails after a command -> local watchdog fallback is switched on and the step
    # is not learned from; no extra remote command is sent next step.
    def test_post_command_readback_failure_activates_local_watchdog(self):
        agent = self.agent()
        original = agent.gateway.execute
        def execute(action):
            response = original(action)
            agent.gateway.unreadable_polls = 2
            return response
        agent.gateway.execute = execute
        event = agent.step(training=True)
        self.assertTrue(event['local_fallback'])
        self.assertFalse(event['learning_updated'])
        agent.step()
        self.assertEqual(agent.gateway.command_count, 1)

    # Q-learning maths: the future value only counts ALLOWED actions (the 1000 on a
    # disallowed action is ignored) and is 0 at the end of an episode.
    def test_q_update_respects_mask_and_terminal(self):
        self.policy.table[('next',)] = {'KEEP': 0.0, NS_GREEN: 1000, EW_GREEN: 0, PEDESTRIAN_WALK: 0}
        self.policy.update(('s',), 'KEEP', 10, ('next',), ('KEEP',))
        self.assertEqual(self.policy.table[('s',)]['KEEP'], 1)
        self.policy.update(('terminal',), 'KEEP', 10, ('next',), (NS_GREEN,), terminal=True)
        self.assertEqual(self.policy.table[('terminal',)]['KEEP'], 1)

    # Saving then loading the Q-table gives an identical table, and asking about an
    # unseen situation does not add anything to it.
    def test_model_round_trip_and_readonly_inference(self):
        self.teach('KEEP')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q.json'
            self.policy.save(path)
            loaded = QLearningAgent.load(path, self.c)
            self.assertEqual(loaded.table, self.policy.table)
            snapshot = repr(loaded.table)
            loaded.choose(('unseen',), ('KEEP',))
            self.assertEqual(repr(loaded.table), snapshot)

    # Working memory keeps only 20 records, but the SQLite database keeps all 25 and
    # still has them (and counts 1 previous run) after closing and reopening.
    def test_memory_persists_and_working_memory_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.sqlite'
            memory = AgentMemory(path)
            for i in range(25):
                memory.record({'time': i, 'action': 'KEEP'})
            self.assertEqual(len(memory.recent), 20)
            memory.close()
            memory = AgentMemory(path)
            self.assertEqual(memory.previous_runs, 1)
            count = memory.connection.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            self.assertEqual(count, 25)
            memory.close()

    # Same seed -> same traffic; different seed -> different traffic.
    def test_arrival_schedules_are_reproducible(self):
        self.assertEqual(make_schedule(50, 100), make_schedule(50, 100))
        self.assertNotEqual(make_schedule(50, 100), make_schedule(51, 100))


    def test_fuzzy_age_changes_permission(self):
        fresh = calculate_fuzzy_reliability(0.90, age_seconds=0, disagreement=0)
        ageing = calculate_fuzzy_reliability(0.90, age_seconds=7, disagreement=0)
        stale = calculate_fuzzy_reliability(0.90, age_seconds=10, disagreement=0)
        self.assertEqual(select_control_mode(fresh), 'Q-learning permitted')
        self.assertEqual(select_control_mode(ageing), 'Re-observe camera')
        self.assertEqual(select_control_mode(stale), 'Safe fallback')
        self.assertGreater(fresh, ageing)
        self.assertGreater(ageing, stale)

    def test_fuzzy_disagreement_overrules_high_confidence(self):
        self.assertEqual(select_control_mode(
            calculate_fuzzy_reliability(0.90, 0, 0.65)), 'Safe fallback')
        older = {'timestamp': 10, 'ns_queue': 2, 'ew_queue': 0}
        current = {'timestamp': 10, 'ns_queue': 17, 'ew_queue': 0}
        self.assertGreater(observation_disagreement(current, older, 10), 0.65)

    def test_ageing_observation_rechecks_and_recovers(self):
        class AgeingCamera:
            def read(self, env, attempt=0):
                return {'camera_available': True, 'camera_confidence': 0.9,
                        'timestamp': env.time - 7 if attempt == 0 else env.time,
                        'ns_queue': len(env.ns), 'ew_queue': len(env.ew),
                        'pedestrian_count': len(env.pedestrians)}
        self.env.time = 20
        agent = TrafficAgent(self.env, AgeingCamera(), self.policy, self.memory)
        event = agent.step()
        self.assertEqual(len(event['observations']), 2)
        self.assertEqual(event['observations'][0]['mode'], 'Re-observe camera')
        self.assertEqual(event['observations'][1]['mode'], 'Q-learning permitted')
        self.assertEqual(event['mode'], 'Safe fallback: unseen state')

    def test_disagreeing_observations_activate_fallback(self):
        self.env.time = 20
        self.env.elapsed = 20
        self.memory.observe({'timestamp': 20, 'ns_queue': 0, 'ew_queue': 0})
        class WrongCamera:
            def read(self, env, attempt=0):
                return {'camera_available': True, 'camera_confidence': 0.9,
                        'timestamp': env.time, 'ns_queue': 15, 'ew_queue': 0,
                        'pedestrian_count': 0}
        event = TrafficAgent(self.env, WrongCamera(), self.policy, self.memory).step()
        self.assertEqual(event['mode'], 'Safe fallback')
        self.assertEqual(len(event['observations']), 1)
        self.assertGreater(event['observations'][0]['disagreement'], 0.6)

    def test_predictive_pedestrian_urgency_respects_minimum_green(self):
        self.env.time = 20
        self.env.elapsed = 14
        self.env.pedestrians.append(0)
        self.assertEqual(self.env.safe_actions(), ('KEEP',))
        self.env.elapsed = 15
        self.assertEqual(self.env.safe_actions(), (PEDESTRIAN_WALK,))

    def test_pedestrian_target_exceedance_is_reported(self):
        self.env.time = 31
        self.env.elapsed = 0  # a mandatory green prevents switching yet
        self.env.pedestrians.append(0)
        self.env.tick()
        metrics = self.env.metrics()
        self.assertGreater(metrics['max_pedestrian_wait_s'], 30)
        self.assertGreater(metrics['pedestrian_wait_over_target_seconds'], 0)
        self.assertEqual(metrics['pedestrians_currently_over_target'], 1)
        self.assertEqual(metrics['safety_violations'], 0)

    # Fuzzy thresholds: 0.70 -> Q-learning, 0.40 -> re-observe, just below 0.40 -> fallback;
    # confidence 0.6 gives reliability 2/3 (matches the worked example in the report guide).
    def test_fuzzy_boundaries(self):
        self.assertEqual(select_control_mode(0.70), 'Q-learning permitted')
        self.assertEqual(select_control_mode(0.40), 'Re-observe camera')
        self.assertEqual(select_control_mode(0.399), 'Safe fallback')
        self.assertAlmostEqual(calculate_fuzzy_reliability(0.6), 2 / 3)


if __name__ == '__main__':
    unittest.main()
