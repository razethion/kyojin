"""Existing Kyojin HTTP handlers with the real synthetic cluster request engine."""
import asyncio
import json
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from test_multinode_generation_cpu import engine
from test_multinode_qwen_engine_cpu import Tokenizer
from tools.multinode.qwen_engine import QwenClusterEngine
from tools.multinode.serve_qwen import create_app, load_http

TEMPLATE = '{% for m in messages %}{{m.content}}{% endfor %}'


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.core, self.stages = engine()
        self.engine = QwenClusterEngine(self.core, Tokenizer(), [])
        self.client = TestClient(TestServer(create_app(self.engine, 'cluster-test', TEMPLATE)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    def body(self, **extra):
        return dict(model='cluster-test', messages=[{'role': 'user', 'content': 'hello'}],
                    max_tokens=13, return_token_ids=True, **extra)

    async def test_nonstream_and_stream_tokens_usage_identical(self):
        response = await self.client.post('/v1/chat/completions', json=self.body())
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data['choices'][0]['message']['content'], 'GHIJKLMNOPQRS')
        self.assertEqual(data['choices'][0]['finish_reason'], 'length')
        self.assertEqual(data['usage']['completion_tokens'], 13)
        response = await self.client.post('/v1/chat/completions', json=self.body(stream=True))
        text = await response.text()
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        visible = ''.join(event['choices'][0]['delta'].get('content', '') for event in events)
        self.assertEqual(visible, data['choices'][0]['message']['content'])
        self.assertEqual(events[-1]['token_ids'], data['token_ids'])
        self.assertIn('data: [DONE]', text)
        self.assertFalse(self.core.pipeline.sessions.items)

    async def test_two_concurrent_http_requests(self):
        async def run():
            response = await self.client.post('/v1/chat/completions', json=self.body())
            self.assertEqual(response.status, 200)
            return (await response.json())['choices'][0]['message']['content']
        self.assertEqual(await asyncio.gather(run(), run()), ['GHIJKLMNOPQRS'] * 2)
        self.assertFalse(self.core.pipeline.sessions.items)

    async def test_stream_reports_engine_token_counts(self):
        # Exercise the upstream live= caller without a wall-clock sleep.
        with patch.object(load_http().serve_compat.LiveTimings, 'PERIOD_S', 0.0):
            response = await self.client.post('/v1/chat/completions', json=self.body(stream=True))
            self.assertEqual(response.status, 200)
            text = await response.text()
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        progressive = [event['timings'] for event in events
                       if event.get('timings') and event['choices'][0]['finish_reason'] is None]
        self.assertTrue(progressive)
        counts = [t['predicted_n'] for t in progressive]
        self.assertEqual(counts, sorted(counts))
        self.assertTrue(all(8 <= n <= 13 for n in counts))
        self.assertEqual(events[-1]['timings']['predicted_n'], 13)
        self.assertEqual(events[-1]['usage']['completion_tokens'], 13)

    async def test_tool_suppression_is_rejected_before_stream_admission(self):
        for stream in (False, True):
            response = await self.client.post('/v1/chat/completions',
                                             json=self.body(no_tool_call=True, stream=stream))
            self.assertEqual(response.status, 400)
            self.assertIn('no_tool_call', (await response.json())['error']['message'])
            self.assertFalse(self.core.pipeline.sessions.items)
        response = await self.client.post('/v1/chat/completions', json=self.body(no_tool_call=False))
        self.assertEqual(response.status, 200)
        self.assertFalse(self.core.pipeline.failed)

    async def test_stop_string_and_rejected_sampling_do_not_poison(self):
        response = await self.client.post('/v1/chat/completions', json=self.body(stop=['NOP']))
        self.assertEqual((await response.json())['choices'][0]['message']['content'], 'GHIJKLM')
        response = await self.client.post('/v1/chat/completions', json=self.body(temperature=.8, stream=True))
        self.assertEqual(response.status, 400)
        self.assertFalse(self.core.pipeline.failed)
        response = await self.client.post('/v1/chat/completions', json=self.body())
        self.assertEqual(response.status, 200)

    async def test_models_health_and_failed_worker_health(self):
        response = await self.client.get('/v1/models')
        data = await response.json()
        self.assertEqual(data['data'][0]['max_context'], 256)
        response = await self.client.get('/health')
        data = await response.json()
        self.assertTrue(data['clustered'] and data['greedy_only'])
        self.core.pipeline.close()
        response = await self.client.get('/health')
        self.assertEqual(response.status, 503)


if __name__ == '__main__':
    unittest.main(verbosity=2)
