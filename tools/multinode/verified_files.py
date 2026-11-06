"""Reuse a completed SHA256 verification only while every file identity matches.

The seal is local, private, and created after a successful full verification.
Changing contents, inode, size, mtime, ctime, root or manifest invalidates it.
"""
import hashlib
import json
import os
from pathlib import Path
from . import identity


def model_id(expected):
    return hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def snapshot(root):
    rows=[]
    for p in identity.files(root):
        s=p.stat()
        rows.append(dict(name=p.name,device=s.st_dev,inode=s.st_ino,size=s.st_size,
                         mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
    return rows


def seal(root, manifest_path, receipt_path, verified_model_id):
    expected=json.loads(Path(manifest_path).read_text())
    if model_id(expected)!=verified_model_id: raise ValueError('verification proof model differs')
    rows=snapshot(root)
    if [(r['name'],r['size']) for r in rows] != [(r['name'],r['size']) for r in expected['files']]:
        raise ValueError('file set differs from completed verification')
    p=Path(receipt_path)
    record=dict(version=1,root=str(Path(root).resolve()),model_id=verified_model_id,files=rows,
                proof='completed full SHA256 verification and matched two-node model manifests')
    with p.open('x') as out:
        os.chmod(p,0o600);json.dump(record,out,indent=2)


def verify(root,manifest_path,receipt_path):
    expected=json.loads(Path(manifest_path).read_text());wanted=model_id(expected)
    p=Path(receipt_path)
    try:
        st=p.stat()
        if st.st_uid!=os.getuid() or st.st_mode & 0o077: raise ValueError('untrusted verification receipt')
        r=json.loads(p.read_text())
        if (r.get('version')==1 and r.get('root')==str(Path(root).resolve()) and
                r.get('model_id')==wanted and r.get('files')==snapshot(root)):
            print('Reusing completed SHA256 verification: all file identities unchanged',flush=True)
            return wanted
    except (OSError,ValueError,TypeError): pass
    return identity.verify(root,manifest_path)
