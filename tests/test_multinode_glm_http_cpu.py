"""GLM HTTP adapter, fixed drafting, admission geometry and native parsing."""
import asyncio
import json
import unittest
from unittest.mock import patch
import torch

from aiohttp.test_utils import TestClient, TestServer
from test_multinode_generation_cpu import engine
from test_multinode_qwen_engine_cpu import Tokenizer as BaseTokenizer
from tools.multinode.serve_glm import GlmClusterEngine, create_app, load_http
from tools.multinode.worker import StageClient
from test_multinode_cpu import worker, CachedToy


class Tokenizer(BaseTokenizer):
    def encode(self, text, **kwargs):
        if text and text.isascii() and text.isupper() and text.isalpha():
            return torch.tensor([[ord(c) - 65 for c in text]])
        return super().encode(text, **kwargs)


class GlmHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.core, self.stages = engine()
        self.core.adaptive_drafting = False
        previous = self.stages[-1].call
        def checked(op, args):
            if op == 'draft':
                self.assertFalse(args['adaptive'])
            return previous(op, args)
        self.stages[-1].call = checked
        adapter = GlmClusterEngine(self.core, Tokenizer(), [])
        self.client = TestClient(TestServer(create_app(adapter, 'glm-test', '{{messages[0].content}}')))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    def body(self, **kw):
        return dict(model='glm-test', messages=[dict(role='user', content='hello')], max_tokens=13, **kw)

    async def test_concurrent_requests_fixed_draft_and_geometry(self):
        async def run():
            r = await self.client.post('/v1/chat/completions', json=self.body())
            self.assertEqual(r.status, 200)
            d = await r.json()
            self.assertEqual(d['usage']['completion_tokens'], 13)
            return d['choices'][0]['message']['content']
        self.assertEqual(await asyncio.gather(run(), run()), ['GHIJKLMNOPQRS'] * 2)
        d = await (await self.client.get('/v1/models')).json()
        self.assertEqual(d['data'][0]['max_context'], 256)
        self.assertEqual(d['data'][0]['cache_tokens'], 512)
        self.assertFalse(self.core.pipeline.sessions.items)

    async def test_refusal_then_sse_reuses_sessions(self):
        r = await self.client.post('/v1/chat/completions', json=self.body(temperature=.8))
        self.assertEqual(r.status, 400)
        r = await self.client.post('/v1/chat/completions', json=self.body(stream=True))
        self.assertEqual(r.status, 200)
        self.assertIn('data: [DONE]', await r.text())
        self.assertFalse(self.core.pipeline.failed)
        self.core.pipeline.close()
        self.assertEqual((await self.client.get('/health')).status, 503)

    async def test_stream_reports_engine_token_counts(self):
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

    def test_native_glm_tool_parser(self):
        d = load_http().parse_completion('<think>check</think><tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>')
        self.assertEqual(d['reasoning_content'], 'check')
        self.assertEqual(d['tool_calls'][0]['function']['name'], 'weather')
        self.assertIn('Paris', d['tool_calls'][0]['function']['arguments'])

    def test_authenticated_ping_preserves_live_session(self):
        stage = CachedToy(0, 2)
        with worker(stage) as (port, token, _):
            client = StageClient('127.0.0.1', port, token)
            try:
                opened = client.call('open', {'session': 'live'})
                self.assertEqual(client.ping(), {'alive': True})
                self.assertEqual(stage.sessions.items['live'].length, 0)
                self.assertEqual(stage.ops, ['open'])
                self.assertEqual(opened['slot'], 0)
                client.call('release', {'session': 'live'})
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
