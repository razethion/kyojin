"""Transfer only this pinned public model over the private direct USB4 link."""
import argparse
import concurrent.futures
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import time
import threading
import urllib.request

MODEL = Path('/home/razeth/models/deepseek41-flash-1017-exl3-2.68-608c237')
REV = '608c237e95188739612e435a935ffa614b9c40e9'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(16 << 20): h.update(b)
    return h.hexdigest()


def server(state):
    meta = json.loads((state/'repo-metadata.json').read_text())
    rows = []
    expected = [r for r in meta['siblings'] if '/' not in r['rfilename'] and not r['rfilename'].startswith('.')]
    manifest = dict(revision=REV, files=[], complete=False)
    allowed = set()
    def verify_completed():
        try:
            while True:
                for row in expected:
                    name=row['rfilename'];p=MODEL/name
                    if name in allowed or not p.is_file(): continue
                    size=row.get('size',p.stat().st_size)
                    if p.stat().st_size != size: raise RuntimeError('size differs '+name)
                    digest=sha(p);wanted=(row.get('lfs') or {}).get('sha256')
                    if wanted and wanted!=digest: raise RuntimeError('LFS hash differs '+name)
                    rows.append(dict(name=name,size=size,sha256=digest));allowed.add(name)
                    print(json.dumps(dict(verified=name,size=size)),flush=True)
                manifest['files']=list(rows)
                manifest['complete']=len(rows)==len(expected) and (state/'download-complete.json').exists()
                (state/'public-transfer-manifest.json').write_text(json.dumps(manifest,indent=2))
                if manifest['complete']:
                    from tools.multinode.identity import files
                    selected={p.name for p in files(MODEL)}
                    native=dict(version=1,files=sorted([r for r in rows if r['name'] in selected],key=lambda r:r['name']))
                    (state/'integrity.json').write_text(json.dumps(native,indent=2));return
                time.sleep(10)
        except BaseException as e:
            (state/'transfer-error.json').write_text(json.dumps(dict(error=repr(e))));os._exit(1)
    threading.Thread(target=verify_completed,daemon=True).start()
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.removeprefix('/')
            if name == '_manifest':
                encoded=json.dumps(manifest).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(encoded)));self.end_headers()
                self.wfile.write(encoded);return
            if name not in allowed:
                self.send_error(404);return
            p = MODEL/name
            self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
            with p.open('rb') as f: shutil.copyfileobj(f,self.wfile,4 << 20)
        def log_message(self,*a): pass
    print('PUBLIC_MODEL_TRANSFER_READY', flush=True)
    http.server.ThreadingHTTPServer(('10.203.0.1',9804),Handler).serve_forever()


def client(state):
    state.mkdir(parents=True,exist_ok=True); MODEL.mkdir(parents=True,exist_ok=True)
    url = 'http://10.203.0.1:9804/'
    deadline = time.monotonic()+4*3600
    while True:
        try:
            with urllib.request.urlopen(url+'_manifest',timeout=15) as f: manifest=json.load(f)
            break
        except OSError:
            if time.monotonic()>deadline: raise
            time.sleep(10)
    assert manifest['revision']==REV
    assert shutil.disk_usage(MODEL).free > 395339893915 + 100*1024**3
    started=time.time()
    def fetch(row):
        name=row['name'];assert '/' not in name and not name.startswith('.')
        p=MODEL/name
        if p.is_file() and p.stat().st_size==row['size'] and sha(p)==row['sha256']: return
        part=p.with_suffix(p.suffix+'.usb4-partial')
        h=hashlib.sha256();n=0
        with urllib.request.urlopen(url+name,timeout=180) as f, part.open('wb') as out:
            while b:=f.read(8 << 20):out.write(b);h.update(b);n+=len(b)
        assert n==row['size'] and h.hexdigest()==row['sha256'],name
        part.replace(p)
        print(json.dumps(dict(copied=name,bytes=n)),flush=True)
    copied=set()
    while True:
        rows=manifest['files']
        new=[r for r in rows if r['name'] not in copied]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,new))
        copied.update(r['name'] for r in new)
        if manifest['complete']:break
        if time.monotonic()>deadline:raise TimeoutError('download/copy deadline exceeded')
        time.sleep(10)
        with urllib.request.urlopen(url+'_manifest',timeout=15) as f:manifest=json.load(f)
        assert manifest['revision']==REV
    from tools.multinode.identity import files
    selected={p.name for p in files(MODEL)}
    native=dict(version=1,files=sorted([r for r in rows if r['name'] in selected],key=lambda r:r['name']))
    (state/'integrity.json').write_text(json.dumps(native,indent=2))
    result=dict(revision=REV,files=len(rows),bytes=sum(r['size'] for r in rows),seconds=time.time()-started,
                hashes_verified=True,completed=time.time())
    (state/'copy-complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['server','client']);ap.add_argument('--state',required=True)
    a=ap.parse_args();(server if a.role=='server' else client)(Path(a.state))
