from concurrent.futures import ThreadPoolExecutor

def prefill_pipeline(pipe,name,chunks, *, cancelled=lambda: False, state_only=False,
                     prefetch_ahead=False, hid_rows=None, return_completed=False):
    """Bounded prefill window; every rank preserves original chunk shapes/order."""
    session = pipe.sessions.items[name]
    step, past = session.step, session.length
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    previous = None
    completed = 0
    def remote(args):
        remote_args = dict(args)
        if 'ahead_ids' in remote_args:
            remote_args['ahead_ids'] = None
        out = pipe.clients[1].call('prepare',remote_args)
        got = pipe.clients[1].call('commit',dict(session=name,step=args['step'],accepted=args['input_ids'].shape[1]))
        assert got == {'length':args['past']+args['input_ids'].shape[1],'step':args['step']+1}
        return out
    def complete(args):
        pipe.sessions.begin(name,args['step'],args['past'],args['input_ids'].shape[1])
        pipe.sessions.mark_ready(name)
        pipe.sessions.commit(name,args['step'],args['input_ids'].shape[1])
    try:
        for i,ids in enumerate(chunks):
            if cancelled():
                break
            args = dict(session=name,step=step+i,past=past,input_ids=ids,activation=None,
                        last_tokens_only=1,speculative=False,return_logits=False)
            if state_only or prefetch_ahead or hid_rows is not None:
                args.update(state_only=state_only and i+1 < len(chunks),
                            ahead_ids=chunks[i+1] if prefetch_ahead and i+1 < len(chunks) else None,
                            hid_rows=hid_rows)
            activation = pipe.clients[0].call('prepare',args)
            got = pipe.clients[0].call('commit',dict(session=name,step=step+i,accepted=ids.shape[1]))
            assert got == {'length':past+ids.shape[1],'step':step+i+1}
            if pending is not None:
                pending.result(); complete(previous)
            args['activation'] = activation
            pending = pool.submit(remote,args)
            previous = args; past += ids.shape[1]
            completed += 1
        output = None
        if pending is not None:
            output = pending.result(); complete(previous)
        return (output, completed) if return_completed else output
    except BaseException:
        if pending is not None:
            try: pending.result(timeout=30)
            except BaseException: pass
        pipe.close()
        raise
    finally:
        pool.shutdown(wait=True)

def verify_jobs(pipe, jobs, *, return_logits=False):
    """Pipeline independent sessions' already-drafted blocks; commit afterward."""
    pending = None
    previous = None
    results = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            for name,ids in jobs:
                st=pipe.sessions.items[name]
                pipe.sessions.begin(name,st.step,st.length,ids.shape[1])
                args=dict(session=name,step=st.step,past=st.length,input_ids=ids,
                          activation=None,last_tokens_only=ids.shape[1],speculative=True,return_logits=return_logits)
                activation=pipe.clients[0].call('prepare',args)
                if pending is not None:
                    results[previous]=pending.result();pipe.sessions.mark_ready(previous)
                args['activation']=activation
                pending=pool.submit(pipe.clients[1].call,'prepare',args);previous=name
            results[previous]=pending.result();pipe.sessions.mark_ready(previous)
            return results
        except BaseException:
            if pending is not None:
                try:pending.result(timeout=30)
                except BaseException:pass
            pipe.close();raise


def draft_verify_jobs(pipe, requests):
    """B draft1 -> (A verify1 || B draft2) -> (B verify1 || A verify2).

    One B executor owns every remote frame. Whole-block journals remain pending
    until the caller makes the same accepted-prefix decisions as verify_jobs.
    Caller owns the global engine lock and cancels/commits both sessions.
    """
    if len(requests) != 2 or requests[0].name == requests[1].name:
        raise ValueError('two distinct sessions required')
    if type(pipe.depth) is not int or not 1 <= pipe.depth <= 7:
        raise ValueError('supported paired depth is 1..7')
    import torch
    jobs = []; proposals = {}; results = {}; pending = None
    def draft(r):
        return pipe.draft(r.name, r.current, pipe.depth, adaptive=False)['tokens']
    def args_for(r, proposal):
        ids = torch.cat((r.current, proposal), dim=1)
        st = pipe.sessions.items[r.name]
        pipe.sessions.begin(r.name, st.step, st.length, ids.shape[1])
        jobs.append((r.name, ids)); proposals[r.name] = proposal
        return dict(session=r.name, step=st.step, past=st.length,
                    input_ids=ids, activation=None, last_tokens_only=ids.shape[1],
                    speculative=True, return_logits=getattr(r, "sampler", None) is not None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first, second = requests
            a = args_for(first, draft(first))
            pending = pool.submit(draft, second)
            a['activation'] = pipe.clients[0].call('prepare', a)
            b = args_for(second, pending.result())
            pending = pool.submit(pipe.clients[1].call, 'prepare', a)
            b['activation'] = pipe.clients[0].call('prepare', b)
            results[first.name] = pending.result()
            pipe.sessions.mark_ready(first.name)
            pending = pool.submit(pipe.clients[1].call, 'prepare', b)
            results[second.name] = pending.result()
            pipe.sessions.mark_ready(second.name)
            return jobs, proposals, results
        except BaseException:
            if pending is not None:
                try: pending.result(timeout=30)
                except BaseException: pass
            pipe.close()
            raise
