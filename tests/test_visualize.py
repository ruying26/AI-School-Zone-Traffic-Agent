"""Visual replay must preserve logged AI decisions and work offline."""
import json
import tempfile
import unittest
from pathlib import Path

from visualize import create_dashboard, read_events


class VisualReplayTests(unittest.TestCase):
    def setUp(self):
        self.event = {'state_after': {'time': 5, 'phase': 'NS_GREEN',
                                      'ns_queue': 2, 'ew_queue': 3},
                      'mode': 'Q-learning permitted', 'action': 'KEEP',
                      'observations': [{'score': 0.9}], 'metrics': {}}

    def test_read_logged_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            logfile = Path(temp) / 'execution.jsonl'
            logfile.write_text(json.dumps(self.event) + '\n', encoding='utf-8')
            self.assertEqual(read_events(logfile), [self.event])

    def test_dashboard_contains_self_contained_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            path = create_dashboard(Path(temp) / 'traffic_dashboard.html',
                                    {'normal': [self.event]}, seed=123)
            result = path.read_text(encoding='utf-8')
            self.assertIn('School Zone AI Traffic Controller', result)
            self.assertIn('"seed": 123', result)
            self.assertIn('Q-learning permitted', result)
            self.assertIn('id="qChart"', result)
            self.assertIn('id="road"', result)
            self.assertNotIn('__DATA_JSON__', result)
            self.assertNotIn('<script src=', result)

    def test_dashboard_protects_embedded_json_markup(self):
        evil = dict(self.event, reason='</script><b>test</b>')
        with tempfile.TemporaryDirectory() as temp:
            page = create_dashboard(Path(temp) / 'diagram.html', {'normal': [evil]})
            result = page.read_text(encoding='utf-8')
            self.assertNotIn('</script><b>', result)
            self.assertIn('\\u003c/script\\u003e', result)

    def test_reject_invalid_event_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'execution.jsonl'
            log.write_text('{"hello": "world"}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_events(log)


if __name__ == '__main__':
    unittest.main()
