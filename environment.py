"""One-second school-zone simulator with explicit clearance phases.

The local controller retains timing and the physical crossing request even if
camera observations or remote acknowledgements fail. All durations are classroom
simulation assumptions, not traffic-engineering recommendations.
"""
from collections import deque
import random

NS_GREEN = 'NS_GREEN'
EW_GREEN = 'EW_GREEN'
PEDESTRIAN_WALK = 'PEDESTRIAN_WALK'
KEEP = 'KEEP'
TRANSITIONS = ('YELLOW', 'ALL_RED', 'PED_CLEARANCE')


def make_schedule(seed, seconds, scenario='normal'):
    """Pre-generate exogenous arrivals; controllers receive identical demand."""
    rng = random.Random(seed)
    result = []
    for second in range(seconds):
        ns, ew, ped = 0.26, 0.16, 0.045
        if scenario == 'rush':
            ns, ew, ped = 0.42, 0.29, 0.12
        elif scenario == 'shift':
            ns, ew = (0.34, 0.10) if second < seconds // 2 else (0.10, 0.34)
        result.append((int(rng.random() < ns), int(rng.random() < ew),
                       int(rng.random() < ped)))
    return result


class TrafficEnvironment:
    def __init__(self, config, schedule):
        self.config = config
        self.schedule = schedule
        self.time = 0
        self.phase = NS_GREEN
        self.elapsed = 0
        self.target = None
        self.last_vehicle_phase = EW_GREEN
        self.ns = deque()
        self.ew = deque()
        self.pedestrians = deque()
        self.crossing = []
        self.vehicle_arrivals = 0
        self.pedestrian_arrivals = 0
        self.vehicles_served = 0
        self.pedestrians_served = 0
        self.vehicle_wait_seconds = 0
        self.pedestrian_wait_seconds = 0
        self.max_ped_wait = 0
        self.pedestrian_wait_over_target_seconds = 0
        self.pedestrians_started_over_target = 0
        self.max_queue = 0
        self.switches = 0
        self.safety_violations = 0

    def state(self):
        return {'ns_queue': len(self.ns), 'ew_queue': len(self.ew),
                'pending_request': bool(self.pedestrians),
                'ped_wait_s': self.time - self.pedestrians[0] if self.pedestrians else 0,
                'phase': self.phase, 'phase_elapsed_s': self.elapsed,
                'time': self.time}

    def _other_direction(self):
        current = self.phase if self.phase in (NS_GREEN, EW_GREEN) else self.last_vehicle_phase
        return EW_GREEN if current == NS_GREEN else NS_GREEN

    def safe_actions(self):
        c = self.config
        if self.phase in TRANSITIONS or self.phase == PEDESTRIAN_WALK:
            return (KEEP,)
        if self.elapsed < c.min_green_seconds:
            return (KEEP,)
        # Plan ahead for the 3 s yellow, 2 s all-red and the next decision interval.
        # This is an urgency trigger, NOT a promise that the walk begins by 30 s:
        # mandatory vehicle green and an existing pedestrian crossing may prevent it.
        transition_lead = c.yellow_seconds + c.all_red_seconds + c.step_seconds
        if (self.pedestrians and
                self.state()['ped_wait_s'] + transition_lead >= c.max_pedestrian_wait_seconds):
            return (PEDESTRIAN_WALK,)
        choices = [self._other_direction()]
        if self.pedestrians:
            choices.append(PEDESTRIAN_WALK)
        if self.elapsed < c.max_green_seconds:
            choices.insert(0, KEEP)
        return tuple(choices)

    def fallback_action(self):
        """Fixed vehicle rotation plus protected pedestrian demand service."""
        allowed = self.safe_actions()
        if PEDESTRIAN_WALK in allowed and self.pedestrians:
            return PEDESTRIAN_WALK
        other = self._other_direction()
        if other in allowed and (self.elapsed >= self.config.fallback_green_seconds or KEEP not in allowed):
            return other
        return KEEP if KEEP in allowed else allowed[0]

    def command(self, action):
        if action not in self.safe_actions():
            return False
        if action == KEEP:
            return True
        self.last_vehicle_phase = self.phase
        self.target = action
        self.phase = 'YELLOW'
        self.elapsed = 0
        self.switches += 1
        return True

    def _start_target(self):
        self.phase = self.target
        self.target = None
        self.elapsed = 0
        if self.phase == PEDESTRIAN_WALK:
            self.pedestrians_started_over_target += sum(
                self.time - arrival > self.config.max_pedestrian_wait_seconds
                for arrival in self.pedestrians)
            self.crossing = list(self.pedestrians)
            self.pedestrians.clear()

    def _advance_phase(self):
        c = self.config
        if self.phase == 'YELLOW' and self.elapsed >= c.yellow_seconds:
            self.phase, self.elapsed = 'ALL_RED', 0
        elif self.phase == 'ALL_RED' and self.elapsed >= c.all_red_seconds:
            self._start_target()
        elif self.phase == PEDESTRIAN_WALK and self.elapsed >= c.walk_seconds:
            self.pedestrians_served += len(self.crossing)
            self.crossing = []
            self.phase, self.elapsed = 'PED_CLEARANCE', 0
        elif self.phase == 'PED_CLEARANCE' and self.elapsed >= c.pedestrian_clearance_seconds:
            self.phase, self.elapsed = 'ALL_RED', 0
            self.target = self._other_direction()

    def tick(self):
        ns, ew, ped = self.schedule[self.time] if self.time < len(self.schedule) else (0, 0, 0)
        self.ns.extend([self.time] * ns)
        self.ew.extend([self.time] * ew)
        self.pedestrians.extend([self.time] * ped)
        self.vehicle_arrivals += ns + ew
        self.pedestrian_arrivals += ped
        self.vehicle_wait_seconds += len(self.ns) + len(self.ew)
        self.pedestrian_wait_seconds += len(self.pedestrians)
        self.pedestrian_wait_over_target_seconds += sum(
            self.time + 1 - arrival > self.config.max_pedestrian_wait_seconds
            for arrival in self.pedestrians)
        if self.pedestrians:
            self.max_ped_wait = max(self.max_ped_wait, self.time + 1 - self.pedestrians[0])
        self.max_queue = max(self.max_queue, len(self.ns) + len(self.ew))
        # No vehicle discharge during WALK, yellow, all-red or pedestrian clearance.
        if (self.elapsed + 1) % self.config.service_interval_seconds == 0:
            queue = self.ns if self.phase == NS_GREEN else self.ew if self.phase == EW_GREEN else None
            if queue:
                queue.popleft()
                self.vehicles_served += 1
        if self.crossing and self.phase in (NS_GREEN, EW_GREEN):
            self.safety_violations += 1
        self.time += 1
        self.elapsed += 1
        self._advance_phase()

    def advance(self, local_fallback=False):
        # A local watchdog can manage safe phases while the remote agent is blocked.
        for _ in range(self.config.step_seconds):
            if local_fallback:
                self.command(self.fallback_action())
            self.tick()

    def metrics(self):
        return {'vehicle_wait_seconds': self.vehicle_wait_seconds,
                'mean_vehicle_delay_so_far_s': self.vehicle_wait_seconds / max(1, self.vehicle_arrivals),
                'pedestrian_wait_seconds': self.pedestrian_wait_seconds,
                'mean_pedestrian_wait_so_far_s': self.pedestrian_wait_seconds / max(1, self.pedestrian_arrivals),
                'max_pedestrian_wait_s': self.max_ped_wait,
                'pedestrian_wait_target_s': self.config.max_pedestrian_wait_seconds,
                'pedestrian_wait_over_target_seconds': self.pedestrian_wait_over_target_seconds,
                'pedestrians_started_over_target': self.pedestrians_started_over_target,
                'pedestrians_currently_over_target': sum(
                    self.time - t > self.config.max_pedestrian_wait_seconds
                    for t in self.pedestrians),
                'vehicle_arrivals': self.vehicle_arrivals, 'vehicles_served': self.vehicles_served,
                'pedestrian_arrivals': self.pedestrian_arrivals, 'pedestrians_served': self.pedestrians_served,
                'vehicles_remaining': len(self.ns) + len(self.ew),
                'pedestrians_waiting': len(self.pedestrians), 'pedestrians_crossing': len(self.crossing),
                'max_queue': self.max_queue, 'switches': self.switches,
                'safety_violations': self.safety_violations}
