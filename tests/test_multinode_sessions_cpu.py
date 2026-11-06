"""Session ownership and transactional position tests; no torch/GPU required."""
import unittest
from tools.multinode.session import Sessions

class SessionTests(unittest.TestCase):
    def test_disjoint_slots_and_safe_reuse(self):
        state=Sessions(512,2)
        self.assertEqual(state.open('a')['slot'],0)
        self.assertEqual(state.open('b')['slot'],1)
        with self.assertRaises(ValueError):state.open('c')
        state.release('a')
        self.assertEqual(state.open('c')['slot'],0)
        self.assertEqual(state.items['c'].length,0)

    def test_prepare_does_not_commit_and_replay_is_rejected(self):
        state=Sessions(256,1);state.open('a')
        state.begin('a',0,0,17)
        self.assertEqual(state.items['a'].length,0)
        for action in (lambda:state.begin('a',0,0,17),lambda:state.commit('a',0),lambda:state.release('a')):
            with self.assertRaises(ValueError):action()
        state.mark_ready('a')
        self.assertEqual(state.commit('a',0),{'length':17,'step':1})
        with self.assertRaises(ValueError):state.commit('a',0)
        with self.assertRaises(ValueError):state.begin('a',0,0,17)
        with self.assertRaises(ValueError):state.begin('a',1,16,1)
        state.begin('a',1,17,1);state.mark_ready('a');state.commit('a',1)
        self.assertEqual(state.items['a'].length,18)

    def test_capacity_and_types_checked_before_pending_mutation(self):
        state=Sessions(256,1);state.open('a')
        for values in ((0,0,257),(True,0,1),(0,0,0),(0,-1,1)):
            with self.assertRaises(ValueError):state.begin('a',*values)
            self.assertIsNone(state.items['a'].pending)
        for name in ('a','../b','',object()):
            with self.assertRaises(ValueError):state.open(name)

    def test_sessions_advance_independently(self):
        state=Sessions(256,2);state.open('a');state.open('b')
        state.begin('a',0,0,7);state.mark_ready('a');state.commit('a',0)
        self.assertEqual(state.items['b'].length,0)
        state.begin('b',0,0,3);state.mark_ready('b');state.commit('b',0)
        self.assertEqual((state.items['a'].length,state.items['b'].length),(7,3))

if __name__=='__main__':unittest.main(verbosity=2)
