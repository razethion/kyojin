"""Managed DeepSeek cluster process, GPU exclusivity and continuous RAM headroom guard."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('role', choices=('worker', 'coordinator'))
ap.add_argument('--context', type=int, default=262144)
ap.add_argument('--depth', type=int, default=5, choices=range(1,6))
ap.add_argument('--qualify', action='store_true')
ap.add_argument('--optimize',action='store_true')
ap.add_argument('--diagnose',action='store_true')
args = ap.parse_args()
host = os.uname().nodename.split('.')[0]
assert host == ('llama2' if args.role == 'worker' else 'llama')
root = Path('/home/razeth/kyojin-multinode')
setup = Path('/home/razeth/llama2-setup' if host == 'llama2' else '/home/razeth/kyojin-cluster-setup')
state = setup / 'ds41-268-20261102'
state.mkdir(exist_ok=True)
lock = (setup/'qwen605-mtp-20261101/gpu-experiment.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if host == 'llama':
    for unit in ('kyojin-qwen.service','kyojin-glm53.service','kyojin-glm-cluster.service'):
        assert subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode != 0
model = '/home/razeth/models/deepseek41-flash-1017-exl3-2.68-608c237'
if host == 'llama2':
    assert subprocess.run(['systemctl','--user','is-active','--quiet','kyojin-glm-cluster-worker.service']).returncode != 0
profile = json.loads((root/'profiles/ds41-268-cluster.json').read_text())
minimum_free = int(float(profile.get('minimum_free_gib',4))*1024**3)
assert 2*1024**3 <= minimum_free <= 32*1024**3
sys.path.insert(0,str(root))
from tools.multinode.health_watch import HealthWatch
watch = HealthWatch() if args.role == 'coordinator' else None
assert hashlib.sha256((root/'exllamav3_ext.cpython-312-x86_64-linux-gnu.so').read_bytes()).hexdigest() == profile['native_binary_sha256']
env = profile['environment']
env.update(EXL3_DENSE_GEMM_TUNE='0', OMP_NUM_THREADS='4', KYOJIN_SSE_PING_S='15')
for key in ('EXL3_DRAFT2S','EXL3_GR_GS_SIDECAR','EXL3_MTP_EH_FP16','EXL3_MTP_PRENORM_H'):
    env[key] = ''
envpath = state/'engine-env.json'
envpath.write_text(json.dumps(env,indent=2)+'\n')
base = ['--model',model,'--weights-manifest',str(state/'integrity.json'),'--token-file',str(state/'token')]
if args.role == 'worker':
    command = ['.venv/bin/python','-m','tools.multinode.worker',*base,
        '--modules','22:44','--device','cuda:0','--host','10.203.0.2','--port','9034',
        '--usb4-interface','thunderbolt0','--once','--accept-timeout','1800',
        '--cache-context',str(args.context),'--cache-sessions','2','--cache-bits','16',
        '--max-rows',str(profile.get('max_prefill_rows',256)),'--experimental-ds41','--speculation-depth',str(args.depth)]
else:
    command = ['.venv/bin/python','-m','tools.multinode.serve_ds41',*base,
        '--env',str(envpath),'--context',str(args.context),'--sessions','2','--depth',str(args.depth),
        '--chunk',str(profile.get('prefill_chunk',256)),'--max-rows',str(profile.get('max_prefill_rows',256)),'--host','0.0.0.0','--port','8731','--model-id','deepseek-v4.1-flash-1017-2.68bpw']
if args.qualify and args.role == 'coordinator': command += ['--qualify',str(state)]
elif args.role == 'coordinator' and 'requested_draft_depth' in profile:
    command += ['--draft-depth',str(profile['requested_draft_depth'])]
if args.optimize and args.role == 'coordinator':command += ['--optimize',str(state)]
if args.diagnose and args.role == 'coordinator':command += ['--diagnose',str(state)]
script = f'''export EXL3_ROOT={root}
export EXL3_VENV={root}/.venv
export EXL3_ROCM_SDK={root}/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
source tools/strix_halo/env.sh
export PYTHONPATH=.
'''
script += '\n'.join(f'export {k}={shlex.quote(v)}' for k,v in env.items())+'\n'
script += 'exec '+shlex.join(command)+'\n'
attempt = time.time_ns()
logpath = state/f'{args.role}-{attempt}.log'
start = time.monotonic()
process = None
stopping = False
def stop(sig, frame):
    global stopping
    stopping = True
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
minimum = 2**64
failure = None
last_history = 0.
with logpath.open('x') as log, logpath.with_suffix('.memory.jsonl').open('x',buffering=1) as history:
    process = subprocess.Popen(['bash','-c',script],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ready = False
    while process.poll() is None:
        fields = {l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l and len(l.split())>=2}
        available = fields['MemAvailable']
        minimum = min(minimum,available)
        ready = ready or 'cluster_ready' in logpath.read_text(errors='replace') or '"listening"' in logpath.read_text(errors='replace')
        report = dict(role=args.role,pid=process.pid,context=args.context,sessions=2,
            cache_tokens=args.context*2,ready=ready,elapsed_seconds=time.monotonic()-start,
            minimum_available_bytes=minimum,available_bytes=available,
            swap_used_bytes=fields.get('SwapTotal',0)-fields.get('SwapFree',0),log=str(logpath))
        (state/f'{args.role}-status.json').write_text(json.dumps(report)+'\n')
        if time.monotonic()-last_history>=30 or available<minimum_free:
            sample=dict(time=time.time(),**report,minimum_required_available_bytes=minimum_free)
            sample['memory_pressure']=Path('/proc/pressure/memory').read_text().strip()
            sample['gtt_used_bytes']={str(p):int(p.read_text()) for p in Path('/sys/class/drm').glob('card[0-9]/device/mem_info_gtt_used')}
            history.write(json.dumps(sample)+'\n');last_history=time.monotonic()
        unhealthy = watch.check(ready,time.monotonic()) if watch else False
        if available < minimum_free or unhealthy or (not ready and time.monotonic()-start > (2400 if args.qualify else 1200)):
            failure = ('headroom below configured reserve' if available<minimum_free else
                       'coordinator health failed three checks' if unhealthy else 'startup timeout')
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
            break
        time.sleep(2)
    report.update(exit=process.wait(),guard_failure=failure,stopped=stopping)
    (state/f'{args.role}-status.json').write_text(json.dumps(report)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(1 if failure else report['exit'])
