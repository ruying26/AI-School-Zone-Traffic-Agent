# School Zone AI Traffic Agent (5023CMD Coursework 2)

This is a **classroom-only Python simulation** of a four-way school-zone junction.
The AI is a **trained tabular Q-learning policy**, with safety constraints applied before
any simulated traffic-light command. The sensors, API connection, and traffic flow are synthetic.
No external Python packages are needed.

## Quick start (VS Code terminal)

Run commands from inside the `School_Zone_AI_CW2` folder:

```bash
python -m unittest discover -s tests -v
python main.py --scenario all --verbose --every 5
python evaluate.py
```

On Windows, use `py` instead of `python` if necessary.
The supplied `artifacts/q_table.json` is already trained using 1,500 episodes.
To train the model again after modifying the agent or configuration, run `python train.py`.
Training **overwrites** `artifacts/q_table.json` and training evidence.

### Five independent demo scenarios

`python main.py --scenario all --verbose --every 5` runs these one after another:

| Scenario | What the demo changes |
|---|---|
| `normal` | Ordinary vehicle and pedestrian demand |
| `rush` | Higher vehicle and pedestrian arrival probabilities |
| `shift` | More north/south demand at first, then more east/west demand |
| `uncertainty` | Noisy readings, repeated uncertainty, camera outages, and older observations |
| `failures` | Human override, release, lost API acknowledgements, unreadable controller state |

For only one scenario, use `python main.py --scenario uncertainty --verbose` or
`python main.py --scenario shift --verbose`. The `--steps 120` default is
600 seconds of simulated time per scenario, not 600 seconds of real waiting.

Outputs for `--scenario all` are saved separately under
`artifacts/demos/<scenario>/execution.jsonl` and `metrics.json`.
`artifacts/demo_output.txt` contains an example run of all five scenarios.
The test results are in `artifacts/test_results.txt`.

## Interactive traffic junction diagram (recommended for lecturer demo)

A new standard-library-only browser visualization replays the **real Python AI agent log**
so the school-zone lights, queue counts, crossing status and fuzzy decisions are
visible instead of only printed in the terminal. There are no new Python dependencies.

```bash
python visualize.py --scenario all --open
```

If your browser does not open automatically, double-click
`artifacts/traffic_dashboard.html` after running the command. This is a standalone
HTML file and can also be opened offline from the extracted ZIP.

The dashboard provides a four-way junction diagram with changing red/yellow/green
lights, a pedestrian WALK status, illustrative vehicle queues, and an interactive
timeline (Play, Pause, Next, Back, Reset, speed and scenario selection). It also
shows the agent's recorded mode, chosen action, safety-permitted actions, fuzzy
reliability, camera age/disagreement, Q-values, command status, and warnings.
A queue-history graph and the already-computed fixed-time versus AI evaluation
provide evidence for the demo. **The vehicles are icons, not tracked individual cars.**

Useful variations:

```bash
python visualize.py --scenario uncertainty --open
python visualize.py --scenario failures --open
python visualize.py --scenario all --use-saved --open
```

`--use-saved` is only appropriate after the corresponding logs have already been
generated. The visualizer uses fixed random seeds by default, so a rerun with the
same model and settings shows the same decisions. `artifacts/summary.json` contains
the separate ten-seed comparison; the visualization **does not recompute** that
comparison. Run `python evaluate.py` if evaluation settings or model have changed,
then rerun the visualizer. The diagram is a replay, not a physical traffic controller.

## AI design: what happens at each decision

1. The local signal controller reports phase, timing, and pedestrian button state.
2. The synthetic camera reports queue lengths and a confidence value.
3. A validation gate rejects missing, malformed, future-dated, or too-old readings.
4. **Zero-order Sugeno fuzzy logic** grades the observation using camera confidence,
   observation age, and disagreement from a recent valid camera reading.
5. A sufficiently reliable observation permits the Q-learning policy to recommend a phase.
   Medium reliability triggers up to two re-observations; low reliability triggers fallback.
6. The local rules enforce minimum green, maximum green, protected pedestrian phase,
   yellow intervals, and all-red clearance. Human actions cannot bypass these rules.
7. A simulated gateway sends at most one command per decision and reads back its state.
   Unexpected failures activate a local watchdog or report an alert.
8. The agent records rewards and actions in short-term memory and an optional SQLite log.

**Information limits:** disagreement compares consecutive camera readings, allowing for
possible queue changes between samples. There is no second independent camera and
confidence is simulated, not calibrated. Missing sensor data never means zero cars.
The agent uses true simulated queue lengths for training feedback only, not for online
traffic-state input. Re-training is required if rules or settings change materially.

### Fuzzy reliability rules

| Observation age | Low disagreement | Moderate disagreement | High disagreement |
|---|---|---|---|
| Fresh | High permission | Medium permission | Low permission |
| Ageing | Medium permission | Medium permission | Low permission |
| Stale | Low permission | Low permission | Low permission |

Fuzzy membership functions blend neighbouring rows and columns, with rule outputs
high=0.90, medium=0.55 and low=0.20. The camera-confidence fuzzy score caps the
rule result. A final reliability >= 0.70 allows Q-learning; >= 0.40 requests another
observation; lower scores trigger fallback. Sensor validation rejects observations
older than 10 seconds before fuzzy evaluation.

## Pedestrian safety and the 30-second setting

The `max_pedestrian_wait_seconds` value is **an urgency target, not a hard guarantee**.
The controller now anticipates the time required for yellow, all-red, and the next
decision interval, and prioritises a pending pedestrian WALK before it reaches the
30-second threshold whenever the current phase can safely change.

A crossing may still begin after 30 seconds. For example, a new pedestrian who
arrives just after a WALK begins must wait through the rest of that walk and its
clearance, then a protected vehicle green and the next yellow/all-red sequence.
With the simulated durations, that can require around 42 seconds. Allowing an unsafe
immediate light switch to force a 30-second maximum would be incorrect.

Reported metrics now distinguish:

- `max_pedestrian_wait_s`: largest simulated wait observed among queued pedestrians.
- `pedestrian_wait_over_target_seconds`: sum of pedestrian waiting person-seconds beyond the target.
- `pedestrians_started_over_target`: pedestrians who began WALK after the target.
- `pedestrians_currently_over_target`: late pedestrians still waiting at the end.
- `safety_violations`: simulated conflicts detected by the simplified simulation.

These metrics make it possible to discuss the missed target rather than misreport
it as guaranteed safety. Zero simulated conflicts is **not** real-road safety certification.

## Evaluation: AI vs fixed-timer controller

The evaluation uses **10 held-out random seeds per scenario**, identical demand
sequences for both controllers, frozen Q-learning weights, and 120 decisions per run.
Values below come from this updated code and retrained Q-table:

| Scenario | Fixed vehicle wait (s) | AI vehicle wait (s) | Fixed pedestrian wait (s) | AI pedestrian wait (s) | AI max pedestrian wait (s) | AI over-target wait (person-seconds) |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 147.59 | 139.09 | 20.17 | 22.42 | 43.6 | 65.2 |
| Rush | 213.38 | 209.70 | 21.62 | 23.43 | 44.6 | 191.4 |
| Shift | 164.28 | 153.37 | 20.17 | 22.75 | 43.4 | 63.9 |
| Uncertainty | 147.59 | 139.40 | 20.17 | 22.34 | 43.6 | 67.0 |
| Failures | 147.03 | 140.53 | 20.39 | 22.58 | 43.4 | 66.2 |


Both mean waiting-time columns are **cumulative waiting seconds divided by all arrivals**
through the simulation, including arrivals still waiting at the end. They are **not**
completed-trip travel times. The max wait and over-target figure are averages across
10 simulation runs. The AI shows lower average vehicle wait in this evaluation but
slightly higher average pedestrian wait. Both controllers recorded zero simulated
safety violations in these runs. These are results of a synthetic experiment, not
measurements from an actual school junction.

## Main files

| File | Purpose |
|---|---|
| `config.json`, `settings.py` | Validated timing, training, and reward settings |
| `environment.py` | Traffic arrivals, signal sequencing, safety rules, pedestrian metrics |
| `sensors.py` | Synthetic camera, noise/age injection, observation validation |
| `fuzzy_logic.py` | Fuzzy membership, 3x3 rule table and disagreement comparison |
| `controller.py` | AI control loop, human override, command verification and fallback |
| `q_learning.py` | Q-value update, action masking, model save and load |
| `memory.py` | Bounded working memory and SQLite logs |
| `main.py` | Scenario demo and decision trace |
| `visualize.py` | Standalone offline dashboard with junction diagram and queue graphs |
| `train.py` | Reproducible Q-learning training |
| `evaluate.py` | Held-out paired comparison |
| `tests/test_agent.py` | Automated checks (30 tests) |
| `artifacts/q_table.json` | Retrained policy (724 learned states) |
| `artifacts/training.csv`, `artifacts/summary.json` | Training/evaluation evidence |

See `DEMO_GUIDE.md` for a presentation walkthrough and `CHANGELOG_IMPROVEMENTS.md`
for the fixes in this version.
