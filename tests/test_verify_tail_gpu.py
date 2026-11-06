"""Opt-in existing-kernel GPU proof. Run on an isolated GPU, never at collection time.
DSV41_RUN_VERIFY_GPU=1 python -m unittest tests.test_verify_tail_gpu
Full model gate additionally requires identical greedy tokens, logits/taps/routes,
all accepted-prefix rollback states and ring-wrap/compressor boundaries, each flag
alone then combined versus disabled source with identical weights/input/state.
"""
import os
import unittest

@unittest.skipUnless(os.environ.get('DSV41_RUN_VERIFY_GPU') == '1', 'explicit isolated GPU gate required')
class TailProof(unittest.TestCase):
    def test_existing_tail_all_rows_groups(self):
        import torch
        from exllamav3.modules.dsv41_fn import dsv41_fuse as FZ
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(190)
        for rows in range(2,9):
            for groups in range(1,4):
                shared = torch.randn(rows,2560,device='cuda',dtype=torch.half)
                parts = [torch.randn(rows,2560,device='cuda',dtype=torch.float32) for _ in range(groups)]
                # GroupedMoE.run already performs this same ordered FP32 sum.
                routed = parts[0].clone()
                for part in parts[1:]: routed.add_(part)
                expected = (routed + shared.float()).half()
                actual = FZ.moe_tail([routed],shared)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual,expected),(rows,groups))

if __name__ == '__main__': unittest.main()
