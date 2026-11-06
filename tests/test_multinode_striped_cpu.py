"""Two-lane stream framing, pairing and exact tensor round-trip tests."""
import os,socket,threading,unittest
import torch
from tools.multinode.striped import accept_socket,StripedSocket,MAX_PACKET,LENGTH
from tools.multinode.wire import Channel,ProtocolError


class StripedTests(unittest.TestCase):
    def test_exact_large_small_and_empty_frames(self):
        token=os.urandom(32)
        errors=[];counts={}
        with socket.socket() as first,socket.socket() as second:
            for listener in (first,second):listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(10)
            def server():
                try:
                    sock,_=first.accept()
                    stream=accept_socket(sock,second,token,timeout=10)
                    channel=Channel.accept(stream,token,timeout=10)
                    try:
                        for _ in range(4):channel.send(channel.recv())
                        counts['rx']=stream.rx_lane_bytes[:]
                        counts['tx']=stream.tx_lane_bytes[:]
                    finally:channel.close()
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=server);thread.start()
            from tools.multinode.striped import connect_channel
            client=connect_channel(first.getsockname(),second.getsockname(),token,timeout=10)
            try:
                values=[{'small':1},torch.empty((0,4)),torch.arange(1<<20,dtype=torch.int32),torch.arange(2*128*4*256,dtype=torch.float32).reshape(2,128,4,256)]
                for value in values:
                    client.send(value);actual=client.recv()
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,actual))
                    else:self.assertEqual(actual,value)
                self.assertGreater(client.sock.tx_lane_bytes[1],0)
            finally:client.close();thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors,errors)
            self.assertGreater(counts['rx'][1],0)
            self.assertGreater(counts['tx'][1],0)

    def test_authenticated_but_mismatched_join_is_rejected(self):
        token=os.urandom(32);errors=[]
        with socket.socket() as first,socket.socket() as second:
            for listener in (first,second):listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(5)
            def server():
                try:
                    sock,_=first.accept();accept_socket(sock,second,token,timeout=5)
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=server);thread.start()
            channels=[]
            try:
                for lane,listener in enumerate((first,second)):
                    channel=Channel.connect(*listener.getsockname(),token,timeout=5);channels.append(channel)
                    channel.send({'op':'stripe-join-v1','identity':str(lane)*32,'lane':lane,'threshold':65536})
                with self.assertRaises((EOFError,OSError)):channels[0].recv()
            finally:
                for channel in channels:channel.close()
                thread.join(timeout=6)
            self.assertFalse(thread.is_alive())
            self.assertTrue(any(isinstance(error,ProtocolError) for error in errors),errors)

    def test_oversized_striped_packet_rejected_before_allocation(self):
        a,b=socket.socketpair();c,d=socket.socketpair()
        stream=StripedSocket(b,d)
        try:
            a.sendall(LENGTH.pack(MAX_PACKET+1))
            with self.assertRaises(ProtocolError):stream.recv_into(memoryview(bytearray(1)))
        finally:stream.close();a.close();c.close()

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
