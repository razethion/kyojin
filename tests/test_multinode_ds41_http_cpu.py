"""Native DeepSeek request format, output parser and clustered async adapter."""
import asyncio
import unittest
from aiohttp.test_utils import TestClient, TestServer
from test_multinode_generation_cpu import engine
from test_multinode_glm_http_cpu import Tokenizer
from tools.multinode.serve_ds41 import DeepSeekClusterEngine, create_app, load_http


class DeepSeekHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.core, self.stages = engine(); self.core.adaptive_drafting = False
        adapter = DeepSeekClusterEngine(self.core, Tokenizer(), [])
        self.client = TestClient(TestServer(create_app(adapter, 'ds41-test')))
        await self.client.start_server()

    async def asyncTearDown(self): await self.client.close()

    def body(self, **kw):
        return dict(model='ds41-test', messages=[dict(role='user',content='hello')], max_tokens=13,
                    thinking=False, **({'temperature': 0} | kw))

    async def test_concurrent_json_sse_and_geometry(self):
        async def request():
            r = await self.client.post('/v1/chat/completions', json=self.body())
            self.assertEqual(r.status, 200, await r.text())
            d = await r.json()
            self.assertEqual(d['usage']['completion_tokens'], 13)
            return d['choices'][0]['message']['content']
        self.assertEqual(await asyncio.gather(request(),request()), ['GHIJKLMNOPQRS'] * 2)
        r = await self.client.post('/v1/chat/completions', json=self.body(stream=True))
        self.assertEqual(r.status, 200)
        self.assertIn('data: [DONE]', await r.text())
        d = await (await self.client.get('/v1/models')).json()
        self.assertEqual(d['data'][0]['max_context'],256)
        self.assertEqual(d['data'][0]['cache_tokens'],512)
        self.assertFalse(self.core.pipeline.sessions.items)

    async def test_invalid_sampling_refused_before_model_state(self):
        r=await self.client.post('/v1/chat/completions', json=self.body(temperature=-1))
        self.assertEqual(r.status,400)
        self.assertFalse(self.core.pipeline.failed)
        self.core.pipeline.close()
        self.assertEqual((await self.client.get('/health')).status,503)

    def test_native_chat_tool_syntax_is_preserved(self):
        http = load_http()
        prompt = http.ds41_chat.render_chat([{'role':'user','content':'Hello'}], thinking=True,
                                           reasoning_effort='medium')
        self.assertIn('<｜User｜>',prompt)
        self.assertIn('Hello',prompt)
        self.assertIn('<｜Assistant｜>',prompt)


if __name__ == '__main__': unittest.main()
