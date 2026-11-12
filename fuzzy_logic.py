"""Zero-order Sugeno assessment of simulated camera observation reliability.

Inputs:
  confidence: camera's synthetic confidence estimate, 0..1
  age_seconds: seconds elapsed since the image was captured
  disagreement: 0..1 disagreement against a recent, comparable observation

The 3 x 3 rule table grades (fresh/ageing/stale) x
(low/moderate/high disagreement). Camera confidence caps that grade.
Rule outputs are HIGH=0.90, MEDIUM=0.55 and LOW=0.20.
This is a teaching model, not a calibrated safety instrument.
"""
import math

MAX_RECHECKS = 2
RULE_OUTPUTS = {'high': 0.90, 'medium': 0.55, 'low': 0.20}
RELIABILITY_RULES = {
    'fresh':   {'low': 'high',   'moderate': 'medium', 'high': 'low'},
    'ageing':  {'low': 'medium', 'moderate': 'medium', 'high': 'low'},
    'stale':   {'low': 'low',    'moderate': 'low',    'high': 'low'},
}


def clamp(value):
    return max(0.0, min(1.0, value))


def triangular_membership(value, left, peak, right):
    if not left < peak < right:
        raise ValueError('Triangle requires left < peak < right')
    if value <= left or value >= right:
        return 0.0
    if value == peak:
        return 1.0
    return (value - left) / (peak - left) if value < peak else (right - value) / (right - peak)


def low_reliability_membership(value):
    return clamp((0.50 - value) / 0.30)


def high_reliability_membership(value):
    return clamp((value - 0.50) / 0.30)


def confidence_reliability(confidence):
    memberships = {
        'low': low_reliability_membership(confidence),
        'medium': triangular_membership(confidence, 0.20, 0.50, 0.80),
        'high': high_reliability_membership(confidence),
    }
    weight = sum(memberships.values())
    return sum(memberships[k] * RULE_OUTPUTS[k] for k in memberships) / weight


def age_memberships(seconds):
    """Fresh falls away by 9 s, ageing peaks at 7 s, stale begins at 8 s."""
    return {
        'fresh': clamp((9.0 - seconds) / 5.0),
        'ageing': clamp(1.0 - abs(seconds - 7.0) / 4.0),
        'stale': clamp((seconds - 8.0) / 4.0),
    }


def disagreement_memberships(disagreement):
    """Gradual transitions avoid sudden switches at a single difference value."""
    return {
        'low': clamp((0.40 - disagreement) / 0.30),
        'moderate': clamp(1.0 - abs(disagreement - 0.35) / 0.30),
        'high': clamp((disagreement - 0.45) / 0.30),
    }


def context_reliability(age_seconds, disagreement):
    """Fuzzy AND=min, Sugeno weighted average across the nine rules."""
    ages = age_memberships(age_seconds)
    differences = disagreement_memberships(disagreement)
    numerator = denominator = 0.0
    for age_name, age_strength in ages.items():
        for difference_name, difference_strength in differences.items():
            strength = min(age_strength, difference_strength)
            numerator += strength * RULE_OUTPUTS[RELIABILITY_RULES[age_name][difference_name]]
            denominator += strength
    return numerator / denominator if denominator else 0.20


def calculate_fuzzy_reliability(camera_confidence, age_seconds=0, disagreement=0.0):
    """Reliability 0..1; uncertain age or disagreement can veto high confidence.

    Defaults preserve the previous single-camera demonstration API, while the
    controller passes the actual age and observation disagreement.
    """
    values = (camera_confidence, age_seconds, disagreement)
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for x in values):
        raise ValueError('Fuzzy inputs must be finite numbers')
    if not 0 <= camera_confidence <= 1 or age_seconds < 0 or not 0 <= disagreement <= 1:
        raise ValueError('Fuzzy inputs outside their allowed ranges')
    return min(confidence_reliability(camera_confidence),
               context_reliability(age_seconds, disagreement))


def select_control_mode(reliability_score):
    if reliability_score >= 0.70:
        return 'Q-learning permitted'
    if reliability_score >= 0.40:
        return 'Re-observe camera'
    return 'Safe fallback'


def observation_disagreement(current, previous, now, max_gap_seconds=10):
    """Compare two observed queues without ever consulting simulator true queues.

    Between frames traffic can genuinely change. Allow up to one car per second
    per approach plus a 2-car tolerance before treating a difference as suspect.
    Return 0 if there is no recent comparable previous observation.
    """
    if not previous:
        return 0.0
    last_t = previous.get('timestamp')
    this_t = current.get('timestamp')
    if (not isinstance(last_t, (int, float)) or not isinstance(this_t, (int, float))
            or not math.isfinite(last_t) or not math.isfinite(this_t)):
        return 0.0
    gap = this_t - last_t
    if not 0 <= gap <= max_gap_seconds or this_t > now:
        return 0.0
    differences = []
    for key in ('ns_queue', 'ew_queue'):
        a, b = current.get(key), previous.get(key)
        if type(a) is not int or type(b) is not int:
            return 0.0
        unexplained = max(0.0, abs(a - b) - gap - 2)
        differences.append(unexplained / max(5.0, float(a), float(b)))
    return clamp(max(differences))


def resolve_control_mode(confidence_readings):
    """Small standalone demonstration (age=0 and disagreement=0)."""
    for attempt, confidence in enumerate(confidence_readings[:MAX_RECHECKS + 1], start=1):
        score = calculate_fuzzy_reliability(confidence)
        mode = select_control_mode(score)
        print(f'Observation {attempt}: Reliability = {score:.2f}, Mode = {mode}')
        if mode != 'Re-observe camera':
            return mode
    return 'Safe fallback'


if __name__ == '__main__':
    for confidence, age, difference in ((0.90, 0, 0), (0.90, 7, 0),
                                        (0.90, 0, 0.65), (0.90, 10, 0)):
        score = calculate_fuzzy_reliability(confidence, age, difference)
        print(f'confidence={confidence:.2f}, age={age}s, disagreement={difference:.2f} '
              f'=> score={score:.2f}, mode={select_control_mode(score)}')
    resolve_control_mode([0.60, 0.90])
