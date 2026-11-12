"""Bounded working memory and persistent SQLite episode history.

What this file does (in plain words):
  - Gives the agent TWO kinds of memory:
      1) Working (short-term) memory: the last 20 decision records and the last valid
         camera reading. It lives in RAM and is lost when the program ends.
      2) Episodic (long-term) memory: every decision of every run, saved in a SQLite
         database file (agent_memory.sqlite), so history survives a restart.
  - Can export the current run's decisions to a readable text file (execution.jsonl).
"""
from collections import deque   # a list with a maximum length: old items drop off automatically
import json
from pathlib import Path
import sqlite3                  # built-in Python database, stored as a single file
import uuid                     # makes a unique ID for each run


class AgentMemory:
    def __init__(self, path=None, config=None, metadata=None):
        # path=None  -> working memory only (used during training, so 1,500 episodes
        #               do not flood the database).
        # path given -> also store everything in a SQLite file (used by main.py).
        self.recent = deque(maxlen=20)   # WORKING MEMORY: only the 20 most recent decisions
        self.last_observation = None     # the last VALID camera reading
        self.run_id = uuid.uuid4().hex   # unique ID that labels every record of this run
        self.connection = None
        self.previous_runs = 0           # how many earlier runs the database already holds

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)   # create the folder if missing
            self.connection = sqlite3.connect(path)
            # Table 1 'runs': one row per run (its settings and details).
            self.connection.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, config TEXT, metadata TEXT)')
            # Table 2 'events': one row per decision step, linked to its run by run_id.
            self.connection.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT, sim_time INTEGER, payload TEXT)')
            # Count earlier runs BEFORE adding this one (this is the "Previous persisted runs" line).
            self.previous_runs = self.connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
            # Register this run, saving the config and details (seed, scenario, model used).
            self.connection.execute('INSERT INTO runs VALUES (?, ?, ?)',
                                    (self.run_id, json.dumps(config or {}), json.dumps(metadata or {})))
            self.connection.commit()   # commit = actually write the changes to the file

    def observe(self, reading):
        # Remember the latest valid camera reading. dict() makes a copy so later changes
        # elsewhere cannot alter it.
        self.last_observation = dict(reading)

    def record(self, event):
        # Called once per decision step with the full decision record.
        # Converting to JSON text and back makes an independent copy, so later changes to
        # the program's variables cannot rewrite an earlier record.
        # allow_nan=False refuses invalid numbers (NaN/infinity) rather than saving them.
        encoded = json.dumps(event, allow_nan=False)
        self.recent.append(json.loads(encoded))   # working memory (oldest of 20 drops off)
        if self.connection:
            # Long-term memory: save the same record permanently in SQLite.
            self.connection.execute('INSERT INTO events(run_id, sim_time, payload) VALUES (?, ?, ?)',
                                    (self.run_id, event['time'], encoded))
            self.connection.commit()

    def export(self, path):
        # Write this run's decisions to a .jsonl file (one JSON record per line).
        if not self.connection:
            events = list(self.recent)   # no database -> only the last 20 are available
        else:
            # With a database, export EVERY decision of THIS run, in the order they happened.
            events = [json.loads(row[0]) for row in self.connection.execute(
                'SELECT payload FROM events WHERE run_id = ? ORDER BY id', (self.run_id,))]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')

    def close(self):
        # Close the database file cleanly (main.py calls this at the end of a run).
        if self.connection:
            self.connection.close()
            self.connection = None
