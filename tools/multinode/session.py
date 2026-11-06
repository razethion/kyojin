"""Node-local page ownership and prepare/commit state; no GPU handles on wire."""
from dataclasses import dataclass
import re


@dataclass
class Session:
    slot: int
    length: int = 0
    step: int = 0
    pending: tuple | None = None
    ready: bool = False


class Sessions:
    def __init__(self, context, slots):
        if type(context) is not int or context<256 or context%256 or type(slots) is not int or not 1<=slots<=8:
            raise ValueError('invalid fixed-page session geometry')
        self.context,self.slots=context,slots
        self.items={}

    def open(self, name):
        if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name):
            raise ValueError('invalid session identity')
        if name in self.items:raise ValueError('session already exists')
        used={session.slot for session in self.items.values()}
        free=next((i for i in range(self.slots) if i not in used),None)
        if free is None:raise ValueError('no free session slot')
        self.items[name]=Session(free)
        return {'slot':free,'length':0,'step':0}

    def begin(self, name, step, past, rows):
        session=self.items[name]
        if any(type(v) is not int for v in (step,past,rows)) or rows<1:
            raise ValueError('invalid step geometry')
        if session.pending is not None or step!=session.step or past!=session.length:
            raise ValueError('stale, duplicated or uncommitted step')
        if past+rows>self.context:raise ValueError('session context exceeded')
        session.pending=(step,past,rows)
        session.ready=False
        return session

    def mark_ready(self, name):
        session=self.items[name]
        if session.pending is None:raise ValueError('no pending step')
        session.ready=True

    def validate_commit(self, name, step, accepted=None):
        session=self.items[name]
        if type(step) is not int or session.pending is None or session.pending[0]!=step or not session.ready:
            raise ValueError('no matching completed prepare')
        rows=session.pending[2]
        if accepted is None:accepted=rows
        if type(accepted) is not int or not 1<=accepted<=rows:
            raise ValueError('invalid accepted prefix')
        return accepted

    def commit(self, name, step, accepted=None):
        accepted=self.validate_commit(name,step,accepted)
        session=self.items[name]
        session.length+=accepted
        session.step+=1
        session.pending=None
        session.ready=False
        return {'length':session.length,'step':session.step}

    def release(self, name):
        if self.items[name].pending is not None:
            raise ValueError('pending step must poison the pipeline, not recycle its pages')
        del self.items[name]
        return {'released':True}
