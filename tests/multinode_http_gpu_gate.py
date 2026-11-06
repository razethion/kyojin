"""Actual local HTTP requests over a loaded two-host speculative pipeline."""
import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer
from tools.multinode.qwen_engine import QwenClusterEngine
from tools.multinode.serve_qwen import create_app, load_http


async def run_http_gate(core, tokenizer, eos, inputs, reference):
    adapter = QwenClusterEngine(core, tokenizer, eos)
    http = load_http()
    app = create_app(adapter, 'qwen-6.05-cluster-experimental', http.DEFAULT_TEMPLATE.read_text())
    client = TestClient(TestServer(app))
    await client.start_server()
    records = []
    import torch
    expected = [tokenizer.decode(torch.tensor(reference[i]['tokens'][:16], dtype=torch.int64),
                                 decode_special_tokens=True) for i in range(2)]
    def body(index=0, **extra):
        return dict(model='qwen-6.05-cluster-experimental', prompt=inputs[index][0].tolist(),
                    max_tokens=16, temperature=0, **extra)
    async def complete(index=0):
        response = await client.post('/v1/completions', json=body(index))
        data = await response.json()
        assert response.status == 200, data
        assert data['choices'][0]['text'] == expected[index], data
        assert data['usage']['completion_tokens'] == 16
        return data['choices'][0]['text']
    try:
        response = await client.get('/health')
        health = await response.json()
        assert response.status == 200 and health['clustered'] and health['greedy_only']
        response = await client.get('/v1/models')
        assert (await response.json())['data'][0]['max_context'] == 2048
        assert await complete() == expected[0]
        records.append('JSON exact solo token-reference text')
        response = await client.post('/v1/completions', json=body(stream=True))
        raw = await response.text()
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        assert '[DONE]' in raw and ''.join(item['choices'][0]['text'] for item in events) == expected[0]
        records.append('SSE exact JSON text')
        assert await asyncio.gather(complete(0), complete(1)) == expected
        records.append('two concurrent requests exact sequential references')
        response = await client.post('/v1/chat/completions', json=dict(model='qwen-6.05-cluster-experimental',
            messages=[dict(role='user', content='Name one primary color. Answer briefly.')],
            max_tokens=8, return_token_ids=True, enable_thinking=False))
        data = await response.json()
        assert response.status == 200 and len(data['token_ids']) > 0, data
        records.append('native Qwen chat template and JSON reply')
        response = await client.post('/v1/completions', json={**body(), 'temperature': .8})
        assert response.status == 400
        records.append('sampled request rejected without poisoning')
        # Disconnect after the first actual generated text, then require that
        # the HTTP driver's cancellation releases the distributed request.
        response = await client.post('/v1/completions', json={**body(stream=True), 'max_tokens': 200})
        while True:
            line = await asyncio.wait_for(response.content.readline(), 30)
            assert line, 'stream ended before cancellation point'
            if line.startswith(b'data: {'):
                item = json.loads(line[6:])
                if item.get('choices', [{}])[0].get('text'):
                    break
        response.close()
        for _ in range(100):
            if not core.pipeline.sessions.items: break
            await asyncio.sleep(.1)
        assert not core.pipeline.sessions.items, 'disconnected HTTP request retained its slot'
        assert await complete() == expected[0]
        records.append('client disconnect releases state; next request exact reference')
        assert not core.pipeline.sessions.items
        return dict(passed=True, checks=records, health=health)
    finally:
        await client.close()
