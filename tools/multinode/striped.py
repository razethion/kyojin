"""Optional two-socket byte stream for large frames on independent USB4 links.

Physical sockets authenticate their join before pairing; the ordinary Channel
then authenticates every reassembled logical frame. Small messages stay on the
primary lane. No tensor dtype, frame limit, or model semantics change.
"""
import concurrent.futures
import os
import socket
import struct
from .wire import Channel, ProtocolError, exact, MAGIC, MAX_PAYLOAD, MAX_HEADER

THRESHOLD=64*1024
LENGTH=struct.Struct('!Q')
MAX_PACKET=MAX_PAYLOAD+MAX_HEADER+128


class StripedSocket:
    def __init__(self, primary, auxiliary):
        self.lanes=(primary,auxiliary)
        self.family=primary.family
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.pending=bytearray()
        self.offset=0
        self.closed=False
        self.tx_lane_bytes=[0,0]
        self.rx_lane_bytes=[0,0]

    def settimeout(self,value):
        for lane in self.lanes:lane.settimeout(value)

    def setsockopt(self,*args):
        for lane in self.lanes:lane.setsockopt(*args)

    def sendall(self,data):
        if not data:return
        size=len(data)
        if size>MAX_PACKET:raise ProtocolError('striped packet exceeds bound')
        self.lanes[0].sendall(LENGTH.pack(size))
        split=(size+1)//2 if size>=THRESHOLD else size
        if split==size:
            self.lanes[0].sendall(data)
        else:
            view=memoryview(data)
            jobs=[self.pool.submit(self.lanes[0].sendall,view[:split]),
                  self.pool.submit(self.lanes[1].sendall,view[split:])]
            try:
                for job in jobs:job.result()
            except BaseException:self.close();raise
        self.tx_lane_bytes[0]+=split
        self.tx_lane_bytes[1]+=size-split

    def recv_into(self,view):
        if not len(view):return 0
        if self.offset==len(self.pending):
            size=LENGTH.unpack(exact(self.lanes[0],LENGTH.size))[0]
            if not 0<size<=MAX_PACKET:raise ProtocolError('invalid striped packet length')
            split=(size+1)//2 if size>=THRESHOLD else size
            pending=bytearray(size)
            def fill(lane,start,end):
                part=memoryview(pending)[start:end]
                pos=0
                while pos<len(part):
                    count=self.lanes[lane].recv_into(part[pos:])
                    if not count:raise EOFError('striped peer disconnected')
                    pos+=count
            if split==size:fill(0,0,size)
            else:
                jobs=[self.pool.submit(fill,0,0,split),self.pool.submit(fill,1,split,size)]
                try:
                    for job in jobs:job.result()
                except BaseException:self.close();raise
            self.pending=pending
            self.offset=0
            self.rx_lane_bytes[0]+=split
            self.rx_lane_bytes[1]+=size-split
        count=min(len(view),len(self.pending)-self.offset)
        view[:count]=memoryview(self.pending)[self.offset:self.offset+count]
        self.offset+=count
        return count

    def close(self):
        if self.closed:return
        self.closed=True
        for lane in self.lanes:
            try:lane.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            lane.close()
        self.pool.shutdown(wait=True,cancel_futures=True)


def _join(channel, identity, lane):
    channel.send({'op':'stripe-join-v1','identity':identity,'lane':lane,'threshold':THRESHOLD})


def connect_channel(primary,auxiliary,token,timeout=60):
    channels=[]
    stream=None
    try:
        identity=os.urandom(16).hex()
        for lane,address in enumerate((primary,auxiliary)):
            channel=Channel.connect(*address,token,timeout=timeout)
            channels.append(channel)
            _join(channel,identity,lane)
        for channel in channels:
            if channel.recv()!={'joined':identity}:raise ProtocolError('stripe join acknowledgement differs')
        stream=StripedSocket(*(channel.sock for channel in channels))
        if bytes(exact(stream,len(MAGIC)))!=MAGIC:raise ProtocolError('invalid logical striped protocol')
        return Channel(stream,token,bytes(exact(stream,32)))
    except BaseException:
        if stream:stream.close()
        else:
            for channel in channels:channel.close()
        raise


def accept_socket(primary,auxiliary_listener,token,timeout=60):
    channels=[]
    try:
        first=Channel.accept(primary,token,timeout)
        channels.append(first)
        info=first.recv()
        if type(info) is not dict or set(info)!={'op','identity','lane','threshold'}:
            raise ProtocolError('invalid stripe join')
        identity=info['identity']
        if type(identity) is not str or len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):
            raise ProtocolError('invalid stripe identity')
        if info!={'op':'stripe-join-v1','identity':identity,'lane':0,'threshold':THRESHOLD}:
            raise ProtocolError('invalid primary stripe join')
        second_sock,_=auxiliary_listener.accept()
        second=Channel.accept(second_sock,token,timeout)
        channels.append(second)
        if second.recv()!={'op':'stripe-join-v1','identity':identity,'lane':1,'threshold':THRESHOLD}:
            raise ProtocolError('stripe lanes belong to different sessions')
        for channel in channels:channel.send({'joined':identity})
        return StripedSocket(first.sock,second.sock)
    except BaseException:
        for channel in channels:channel.close()
        if not channels:primary.close()
        raise
