"""Managed GLM cluster process, GPU exclusivity and continuous RAM headroom guard."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('role', choices=('worker', 'coordinator'))
ap.add_argument('--context', type=int, default=262144)
args = ap.parse_args()
host = os.uname().nodename.split('.')[0]
assert host == ('llama2' if args.role == 'worker' else 'llama')
root = Path('/home/razeth/kyojin-multinode')
setup = Path('/home/razeth/llama2-setup' if host == 'llama2' else '/home/razeth/kyojin-cluster-setup')
state = setup / 'glm410-serving-20261102'
state.mkdir(exist_ok=True)
lock = (setup/'qwen605-mtp-20261101/gpu-experiment.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if host == 'llama':
    for unit in ('kyojin-qwen.service','kyojin-glm53.service'):
        assert subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode != 0
model = '/home/razeth/models/glm53-flash-exl3-sc4.10-c6b8ac3'
profile = json.loads((root/'profiles/glm410-cluster-k8.json').read_text())
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
        '--modules','26:50','--device','cuda:0','--host','10.203.0.2','--port','9034',
        '--usb4-interface','thunderbolt0','--once','--accept-timeout','1800',
        '--cache-context',str(args.context),'--cache-sessions','2','--cache-bits','8',
        '--max-rows','256','--experimental-glm','--speculation-depth','2']
else:
    command = ['.venv/bin/python','-m','tools.multinode.serve_glm',*base,
        '--env',str(envpath),'--context',str(args.context),'--sessions','2','--depth','2',
        '--chunk','256','--host','0.0.0.0','--port','8731','--model-id','glm-5.3-flash-yamz']
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
with logpath.open('x') as log:
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
        if available < 4*1024**3 or (not ready and time.monotonic()-start > 1200):
            failure = 'headroom below4GiB or startup exceeded1200s'
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
            break
        time.sleep(2)
    report.update(exit=process.wait(),guard_failure=failure,stopped=stopping)
    (state/f'{args.role}-status.json').write_text(json.dumps(report)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(1 if failure else report['exit'])
