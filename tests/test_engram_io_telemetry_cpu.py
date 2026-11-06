import unittest
from unittest.mock import patch
import numpy as np
from exllamav3.modules.dsv41_fn.engram_io import NativeRowReader,native_request_bytes


class NativeTelemetryTests(unittest.TestCase):
    def test_aligned_cross_sector_duplicate_many_rows_and_overflow(self):
        segments=[('a',4090,264,264),('b',0,8,8)]
        rows=np.arange(100000,dtype=np.int64)
        expected=sum(((base+int(row)*stride+size+4095)//4096-(base+int(row)*stride)//4096)*4096
                     for row in rows for _,base,stride,size in segments)
        self.assertEqual(native_request_bytes(rows,segments),expected)
        self.assertEqual(native_request_bytes([0,0],segments),24576)
        self.assertEqual(native_request_bytes([],segments),0)
        with self.assertRaises(OverflowError):native_request_bytes([2**62],[('a',0,8,8)])
        with self.assertRaises(ValueError):native_request_bytes([-1],segments)

    def test_consecutive_calls_add_only_current_request_lengths(self):
        reader=object.__new__(NativeRowReader)
        reader.segments=[('a',4090,264,264),('b',0,8,8)]
        reader.row_bytes=272;reader.reads=0;reader.bytes_read=0
        reader._pool=None;reader.threads=1
        reader._fd=reader._base=reader._stride=reader._nb=None
        class Lib:
            def ds_engram_gather(self,*args):return 0
        with patch('exllamav3.modules.dsv41_fn.engram_io._gather_lib',return_value=Lib()):
            reader.read_rows([0])
            self.assertEqual((reader.reads,reader.bytes_read),(2,12288))
            reader.read_rows([1,1])
            self.assertEqual((reader.reads,reader.bytes_read),(6,28672))


if __name__=='__main__':unittest.main()
