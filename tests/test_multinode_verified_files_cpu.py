import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.multinode import identity,verified_files


class VerifiedFilesTests(unittest.TestCase):
    def test_seal_reuses_hash_but_detects_content_and_manifest_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'model';root.mkdir();(root/'config.json').write_text('{}')
            weights=root/'weights.safetensors';weights.write_bytes(b'verified model bytes')
            manifest=Path(d)/'integrity.json';manifest.write_text(json.dumps(identity.manifest(root)))
            receipt=Path(d)/'verified-stat.json';mid=identity.verify(root,manifest)
            verified_files.seal(root,manifest,receipt,mid)
            with patch.object(identity,'verify',side_effect=AssertionError('unnecessary rehash')):
                self.assertEqual(verified_files.verify(root,manifest,receipt),mid)
            s=weights.stat();weights.write_bytes(b'corrupt! model bytes')
            import os
            os.utime(weights,ns=(s.st_atime_ns,s.st_mtime_ns))
            with self.assertRaises(ValueError):verified_files.verify(root,manifest,receipt)

    def test_seal_requires_matching_full_verification_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'config.json').write_text('{}');(p/'weights.safetensors').write_bytes(b'x')
            m=p/'manifest.json';m.write_text(json.dumps(identity.manifest(p)))
            with self.assertRaises(ValueError):verified_files.seal(p,m,p/'seal.json','wrong')


if __name__=='__main__':unittest.main()
