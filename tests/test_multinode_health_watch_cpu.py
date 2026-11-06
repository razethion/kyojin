import http.server
import json
import threading
import unittest
from tools.multinode.health_watch import HealthWatch


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.code=200
        case=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(case.code);self.end_headers()
                self.wfile.write(json.dumps({'status':'ok' if case.code==200 else 'failed'}).encode())
            def log_message(self,*args):pass
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.watch=HealthWatch(f'http://127.0.0.1:{self.server.server_port}/health',grace=10)

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def test_loading_and_bind_grace_do_not_restart(self):
        self.code=503
        self.assertFalse(self.watch.check(False,100))
        self.assertFalse(self.watch.check(True,101))
        self.assertFalse(self.watch.check(True,110))
        self.assertEqual(self.watch.failures,0)

    def test_closed_engine_triggers_after_three_actual_http_failures(self):
        self.assertFalse(self.watch.check(True,0))
        self.code=503
        self.assertFalse(self.watch.check(True,10))
        self.assertFalse(self.watch.check(True,12))
        self.assertTrue(self.watch.check(True,14))

    def test_healthy_response_resets_transient_failure_streak(self):
        self.watch.check(True,0);self.code=503
        self.assertFalse(self.watch.check(True,11))
        self.assertFalse(self.watch.check(True,13))
        self.code=200;self.assertFalse(self.watch.check(True,15))
        self.code=503
        self.assertFalse(self.watch.check(True,17))
        self.assertFalse(self.watch.check(True,19))
        self.assertTrue(self.watch.check(True,21))
