"""Observe, assess reliability, choose, constrain, execute, verify and remember.

What this file does (in plain words):
  This is the "brain loop" of the project. Every 5 simulated seconds, TrafficAgent.step()
  runs one decision cycle:
     1. Confirm the real signal state (read back from the controller).
     2. Look at the camera and check the reading (fuzzy logic: can we trust it?).
     3. Choose an action: Q-learning if the camera is trusted, safe fallback if not,
        or a human's request if a person has taken over.
     4. Keep only actions the safety rules allow.
     5. Send the command ONCE, then read back to verify what really happened.
     6. Work out the reward, learn from it (training only) and save a record.
  SignalGateway is the simulated remote link to the signal, which can be made to fail.
"""
import math

from fuzzy_logic import (calculate_fuzzy_reliability, select_control_mode,
                         observation_disagreement, MAX_RECHECKS)
from sensors import valid_observation
from q_learning import encode_traffic_state
from environment import KEEP


# The 'remote API' between the agent and the junction's signal controller. It is simulated
# and can be told to fail, so we can test how the agent handles failures.
class SignalGateway:
    """Simulated remote API. Never auto-retry an ambiguous switch command."""
    def __init__(self, env):
        self.env = env
        # Set to 'timeout_before' or 'timeout_after' to make the NEXT command fail (demo and tests).
        self.next_fault = None
        # How many upcoming state read-backs return nothing (simulates a lost connection).
        self.unreadable_polls = 0
        # Counts commands sent, so tests can prove there was no blind retry.
        self.command_count = 0

    # Ask the controller for its real current state. Returns None if it cannot be read.
    def readback(self):
        # Connection 'down': use up one unreadable poll and report nothing.
        if self.unreadable_polls:
            self.unreadable_polls -= 1
            return None
        state = self.env.state()
        # The local timer and crossing button are independent of camera confidence.
        # Only the local controller's own facts (phase, timer, pedestrian button), NOT camera counts.
        return {k: state[k] for k in ('pending_request', 'ped_wait_s', 'phase', 'phase_elapsed_s', 'time')}

    # Send one command to the controller and report what happened.
    def execute(self, action):
        self.command_count += 1
        # Take the pending fault (if any) and clear it, so a fault only affects one command.
        fault, self.next_fault = self.next_fault, None
        # Fault 1: the command never arrived, so the junction is unchanged.
        if fault == 'timeout_before':
            return {'status': 'timeout', 'action': action}
        # The command reaches the junction. The safety rules may still refuse it (returns False).
        accepted = self.env.command(action)
        # Fault 2: the command DID run but the acknowledgement was lost. The agent cannot tell,
        # so it must read back the real state afterwards.
        if fault == 'timeout_after':
            return {'status': 'timeout', 'action': action}
        # Normal result: acknowledged if the junction accepted it, otherwise rejected.
        return {'status': 'acknowledged' if accepted else 'rejected', 'action': action}


# The decision-making agent. It combines the camera, fuzzy logic, the Q-learning policy,
# the safety rules, the gateway and memory.
class TrafficAgent:
    def __init__(self, env, camera, policy, memory, baseline=False):
        # The parts the agent works with: junction, camera, Q-learning policy and memory.
        self.env, self.camera, self.policy, self.memory = env, camera, policy, memory
        # Every command goes through the (simulated) remote API gateway.
        self.gateway = SignalGateway(env)
        # True = behave as the fixed-timer comparison controller (no camera, no AI). Used by evaluate.py.
        self.baseline = baseline
        # Human override state: has a person taken control, and what did they ask for?
        self.manual_active = False
        self.manual_action = KEEP
        # True after a human releases control: one verified fallback step runs before the AI resumes.
        self.resume_pending = False


    # A human supervisor takes control and requests an action. Safety rules still apply.
    def set_override(self, action):
        # Reject action names that do not exist.
        if action not in ('KEEP', 'NS_GREEN', 'EW_GREEN', 'PEDESTRIAN_WALK'):
            raise ValueError('Unknown manual action')
        self.manual_active, self.manual_action = True, action

    # The human hands control back. The agent must verify the state first (resume_pending).
    def release_override(self):
        self.manual_active = False
        self.resume_pending = True

    # Look at the camera, check the reading, and decide how far to trust it.
    # Returns: the control mode, the reading, the fuzzy score and a log of every attempt.
    def _observe(self):
        history = []
        # First look plus up to MAX_RECHECKS (2) re-observations = at most 3 camera reads.
        for attempt in range(MAX_RECHECKS + 1):
            # Take a camera reading.
            reading = self.camera.read(self.env, attempt)
            # Gatekeeper: is the reading present, fresh and well-formed?
            valid = valid_observation(reading, self.env.time)
            # Valid -> fuzzy reliability score (0 to 1) and the mode it implies.
            # Invalid -> no score and straight to safe fallback.
            age = self.env.time - reading['timestamp'] if valid else None
            disagreement = (observation_disagreement(reading, self.memory.last_observation,
                                                     self.env.time) if valid else None)
            score = (calculate_fuzzy_reliability(reading['camera_confidence'], age, disagreement)
                     if valid else None)
            mode = select_control_mode(score) if valid else 'Safe fallback'
            # Log-friendly copy of the reading (NaN/infinity turned into text so it can be saved as JSON).
            logged_reading = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                              for k, v in reading.items()}
            # Record this attempt for the log and the demo output.
            history.append({'attempt': attempt + 1, 'reading': logged_reading,
                            'valid': valid, 'age_seconds': age,
                            'disagreement': disagreement, 'score': score, 'mode': mode})
            # Remember the latest valid reading in working memory.
            if valid:
                self.memory.observe(reading)
            # Stop as soon as the answer is not 'look again'.
            if mode != 'Re-observe camera':
                return mode, reading, score, history
        # Re-checks used up and still unsure -> safe fallback. (Bounded, never endless.)
        return 'Safe fallback', reading, score, history

    # ONE full decision cycle. training=True lets the Q-table learn; epsilon controls random
    # exploration during training; terminal=True marks the last step of a training episode.
    def step(self, training=False, epsilon=0.0, terminal=False):
        # Step 1: snapshot the junction and start the decision record (event) that will be logged.
        before = self.env.state()
        event = {'time': self.env.time, 'state_before': before, 'observations': [],
                 'score': None, 'action': KEEP, 'proposed_action': KEEP,
                 'alert': None, 'learning_updated': False, 'state_source': None}
        # Step 2: confirm the real controller state before doing anything.
        readback = self.gateway.readback()
        # Working variables: the situation label, whether the local watchdog is needed,
        # and whether this step may be used for learning.
        encoded = None
        local_fallback = False
        learning_allowed = False
        # Remember counters before acting, so the reward can measure what changed during this step.
        switches_before = self.env.switches
        vehicle_wait_before = self.env.vehicle_wait_seconds
        ped_wait_before = self.env.pedestrian_wait_seconds

        # Step 3a: cannot confirm the signal state -> send NO command, alert a supervisor and
        # let the local watchdog run the junction.
        if readback is None:
            event.update(mode='Blocked: unconfirmed signal state',
                         reason='No fresh controller readback. No remote command issued.',
                         alert='Supervisor: verify controller connection.', command_status='not_sent')
            local_fallback = True
        else:
            # Step 3b: state confirmed -> decide. First get the actions the safety rules allow right now.
            allowed = self.env.safe_actions()
            event['allowed_actions'] = list(allowed)
            # Priority order: human override > resume verification > fixed-timer baseline >
            # AI (camera + fuzzy logic + Q-learning).
            if self.manual_active:
                mode = 'Manual override'
                proposed = self.manual_action
                # Do the human's request only if the safety rules allow it; otherwise keep the current phase.
                action = proposed if proposed in allowed else (KEEP if KEEP in allowed else self.env.fallback_action())
                reason = 'Authorised manual control; hard timing and crossing constraints still apply.'
                if action != proposed:
                    reason += ' Requested manual action was blocked by safety rules.'
            # A human just released control: one safe fallback step before the AI takes over again.
            elif self.resume_pending:
                self.resume_pending = False
                mode, proposed, action = 'Resume verification', KEEP, self.env.fallback_action()
                reason = 'Fresh controller state confirmed after manual release; one fallback step before autonomous control.'
            # Fixed-timer comparison controller: no camera, no learning.
            elif self.baseline:
                mode = 'Baseline'
                action = proposed = self.env.fallback_action()
                reason = 'Fixed vehicle timing with demand-based protected pedestrian service.'
            else:
                # AI path: look at the camera (with bounded re-checks) and get the fuzzy-based mode.
                mode, reading, score, history = self._observe()
                event.update(observations=history, score=score)
                # Camera trusted: build the situation from the camera's queue counts plus the controller's
                # own readback (phase, timer, pedestrian button).
                if mode == 'Q-learning permitted':
                    observed = dict(readback, ns_queue=reading['ns_queue'], ew_queue=reading['ew_queue'])
                    encoded = encode_traffic_state(observed)
                    event['observed_state'] = observed
                    event['state_source'] = 'current validated camera + local controller'
                    # Only use the Q-table for situations seen in training (during training, everything is allowed).
                    if training or encoded in self.policy.table:
                        # Pick the best allowed action (or a random one while training, depending on epsilon).
                        proposed = action = self.policy.choose(encoded, allowed, epsilon if training else 0.0)
                        # Save the scores of the allowed actions so the log and demo can show why.
                        event['q_values'] = {a: self.policy.values(encoded)[a] for a in allowed}
                        reason = 'Reliable camera; choose among safety-permitted actions using Q-values.'
                        # This step may be used to update the Q-table (only matters when training).
                        learning_allowed = True
                    else:
                        # Situation never seen in training -> do not guess; use the safe fallback.
                        mode = 'Safe fallback: unseen state'
                        action = proposed = self.env.fallback_action()
                        reason = 'State was not learned in training; use explicit fallback rather than claim an informed policy.'
                else:
                    action = proposed = self.env.fallback_action()
                    # Camera not trusted (missing, invalid, low confidence, or re-checks used up):
                    # the action chosen above is the safe fallback.
                    reason = 'Missing/invalid or low-confidence observation, or bounded rechecks exhausted.'
                # Log when the last valid camera reading was taken.
                if self.memory.last_observation:
                    event['last_valid_observation_time'] = self.memory.last_observation['timestamp']
            # Step 4: record the decision, then execute it ONCE through the gateway (never blindly retried).
            event.update(mode=mode, proposed_action=proposed, action=action, reason=reason)
            response = self.gateway.execute(action)
            event['command_status'] = response['status']
            # Step 5: verify by reading back the real state after the command.
            confirmed = self.gateway.readback()
            event['confirmed_state'] = confirmed
            # Cannot verify the outcome -> alert, stop remote switching, use the local watchdog,
            # and do not learn from this step.
            if confirmed is None:
                event['alert'] = 'Command outcome unconfirmed; stop further switching requests and use local watchdog.'
                event['reason'] += ' Readback unavailable after execution.'
                local_fallback = True
                learning_allowed = False
            # Timeout or rejected, but the state is readable -> use the real state, alert,
            # do not retry, and do not learn from this step.
            elif response['status'] != 'acknowledged':
                event['alert'] = 'Command failed or acknowledgement lost; readback obtained, no retry this step.'
                event['reason'] += ' Actual controller state recovered; ambiguous transition excluded from learning.'
                learning_allowed = False

        # Step 6: move the simulation forward one step (the local watchdog acts if needed).
        self.env.advance(local_fallback=local_fallback)
        after = self.env.state()
        # Reward (never positive): penalties for vehicle waiting, pedestrian waiting (weighted)
        # and every signal switch. Higher (closer to 0) is better.
        reward = -(self.env.vehicle_wait_seconds - vehicle_wait_before) / self.env.config.step_seconds
        reward -= self.env.config.pedestrian_wait_weight * (self.env.pedestrian_wait_seconds - ped_wait_before)
        reward -= self.env.config.switch_penalty * (self.env.switches - switches_before)
        # Step 7: learn from this step (training only, and only if the outcome was clear).
        if training and learning_allowed and encoded is not None:
            # Training uses simulated transition feedback as ground truth. At runtime
            # the frozen policy only receives validated observations, never hidden queues.
            self.policy.update(encoded, event['action'], reward, encode_traffic_state(after),
                               self.env.safe_actions(), terminal=terminal)
            event['learning_updated'] = True
        # Step 8: finish the record and save it in memory.
        event.update(state_after=after, reward=reward, local_fallback=local_fallback,
                     pedestrian_target_exceeded=(after['ped_wait_s'] >
                         self.env.config.max_pedestrian_wait_seconds),
                     metrics=self.env.metrics())
        self.memory.record(event)
        return event
