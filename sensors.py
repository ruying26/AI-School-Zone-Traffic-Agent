"""Synthetic camera observations. Confidence is simulated, not a calibrated detector.

What this file does (in plain words):
  - Pretends to be a traffic camera. It looks at the true state of the simulated junction
    and returns a "reading": queue lengths, pedestrian count and a confidence number.
  - In the 'uncertainty' scenario it deliberately produces bad readings (low confidence,
    missing data, noisy counts) so the fuzzy-logic layer has something to deal with.
  - Provides valid_observation(), a gatekeeper that rejects unusable readings
    BEFORE they reach the fuzzy logic or Q-learning.
"""
import math
import random


class Camera:
    def __init__(self, seed=0, scenario='normal'):
        # Own random generator with a fixed seed, so the "noise" is reproducible:
        # the same seed always gives the same noisy readings.
        self.rng = random.Random(seed)
        self.scenario = scenario   # 'normal' = clean camera; 'uncertainty' = faulty camera

    def read(self, env, attempt=0):
        # attempt = 0 for the first look, 1 or 2 for re-observations requested by the fuzzy layer.

        # The camera's behaviour repeats every 40 decision steps (40 x 5 s = 200 s).
        # "slot" says where we are in that repeating cycle.
        slot = (env.time // env.config.step_seconds) % 40
        confidence = 0.90   # default: a good, reliable reading
        available = True    # default: the camera is working

        if self.scenario == 'uncertainty':
            if 8 <= slot < 13:
                # Temporary problem: poor on the first look (0.60) but a re-observation
                # gives a clear 0.90. Demonstrates "re-observe and recover".
                confidence = 0.60 if attempt == 0 else 0.90
            elif 13 <= slot < 18:
                # Persistent problem: still 0.60 on every retry, so retries run out and
                # the agent falls back to the safe controller.
                confidence = 0.60
            elif 18 <= slot < 23:
                # Camera outage: no data at all (counts become None, not zero).
                available, confidence = False, 0.0
            elif 23 <= slot < 28:
                # Very poor reading: confidence 0.10, fuzzy score ~0.20 -> fallback.
                confidence = 0.10
            # All other slots keep the default good reading (0.90).

        # During an ageing episode the first frame is 7 s old, but a retry
        # obtains a fresh frame. This lets the fuzzy age rules be demonstrated.
        sample_time = env.time
        if self.scenario == 'uncertainty' and 28 <= slot < 33 and attempt == 0:
            sample_time = max(0, env.time - 7)

        # Add counting errors only when confidence is low (below 0.8).
        # Noise is a random whole number from -4 to +4 vehicles.
        noise = 0 if confidence >= 0.8 else self.rng.randint(-4, 4)
        state = env.state()   # the TRUE state of the junction

        # Build the reading: the true counts plus noise, never below 0.
        # If the camera is unavailable, counts are None (unknown), NOT 0, so the agent
        # can never confuse "no data" with "no cars".
        return {'camera_available': available, 'camera_confidence': confidence,
                'timestamp': sample_time,                    # when this reading was taken
                'ns_queue': max(0, state['ns_queue'] + noise) if available else None,
                'ew_queue': max(0, state['ew_queue'] - noise) if available else None,
                'pedestrian_count': max(0, len(env.pedestrians) + noise) if available else None}


def valid_observation(reading, now, max_age=10):
    # Gatekeeper: returns True only if the reading is safe to use. Any failure -> False,
    # and the controller then switches to the safe fallback.

    # 1) The camera must be working.
    if not reading.get('camera_available'):
        return False
    c = reading.get('camera_confidence')
    t = reading.get('timestamp')

    # 2) Confidence must be a real number between 0 and 1 (rejects text, NaN, infinity, 1.5...).
    if not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
        return False

    # 3) The reading must be fresh: taken at most 10 s ago and not from the "future".
    if not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= now - t <= max_age:
        return False

    # 4) All three counts must be whole numbers that are not negative.
    return all(type(reading.get(k)) is int and reading[k] >= 0
               for k in ('ns_queue', 'ew_queue', 'pedestrian_count'))
