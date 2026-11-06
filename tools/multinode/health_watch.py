"""Turn a failed-but-listening coordinator into a managed process restart."""
import json
import urllib.request


class HealthWatch:
    def __init__(self,url='http://127.0.0.1:8731/health',grace=10,failures=3):
        self.url,self.grace,self.limit=url,grace,failures
        self.ready_at=None
        self.failures=0

    def check(self,ready,now):
        if not ready:return False
        if self.ready_at is None:self.ready_at=now
        if now-self.ready_at<self.grace:return False
        try:
            with urllib.request.urlopen(self.url,timeout=2) as response:
                healthy=response.status==200 and json.load(response).get('status')=='ok'
        except Exception:
            healthy=False
        self.failures=0 if healthy else self.failures+1
        return self.failures>=self.limit
