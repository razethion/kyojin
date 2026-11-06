"""Opt-in native fixed-page session parity on one GPU before cached network use."""
import json,os,tempfile
from pathlib import Path
os.environ['EXL3_KVWIN']='0'
import torch
from exllamav3 import Cache,Config,Model
from tools.multinode.fixture import create
from tools.multinode.stage import CachedDenseStage
from tools.multinode.wire import pack,unpack


def main():
    torch.manual_seed(191)
    torch.set_num_threads(2)
    operations=[]
    lengths={'a':0,'b':0};steps={'a':0,'b':0}
    # Interleaved sessions, a page boundary, and slot reuse with a fresh prefix.
    for session,rows in [('a',250),('b',16),('a',17),('b',1),('a',1),('b',3)]:
        ids=torch.randint(0,256,(1,rows),dtype=torch.long)
        operations.append((session,steps[session],lengths[session],ids))
        lengths[session]+=rows;steps[session]+=1
    operations.append(('c',0,0,torch.randint(0,256,(1,7),dtype=torch.long)))
    with tempfile.TemporaryDirectory(prefix='cluster-cached-') as temp:
        root=Path(temp);create(root)
        model=Model.from_config(Config.from_directory(str(root)))
        cache=Cache(model,max_num_tokens=1024)
        model.load(device='cuda:0',progressbar=False)
        expected=[]
        with torch.inference_mode():
            for name,step,past,ids in operations:
                slot=1 if name=='b' else 0
                params={'attn_mode':'flash_attn','cache':cache,
                        'block_table':torch.arange(slot*2,(slot+1)*2,dtype=torch.int32).view(1,2),
                        'cache_seqlens':torch.tensor([past],dtype=torch.int32),'last_tokens_only':1}
                output=model.forward(ids,params).cpu()
                assert torch.isfinite(output).all(),(name,step)
                expected.append(output)
        model.unload()
        stages=[]
        for begin,end in ((0,2),(2,5)):
            stage=CachedDenseStage(Model.from_config(Config.from_directory(str(root))),begin,end,device='cuda:0',
                                   model_id='fixture',source_id='same',max_rows=256,context=512,sessions=2)
            stage.load();stages.append(stage)
        assert set(stages[0].cache.layers)=={(0,0)}
        assert set(stages[1].cache.layers)=={(1,0)}
        try:
            for stage in stages:
                stage.execute('open',{'session':'a'});stage.execute('open',{'session':'b'})
            for index,(name,step,past,ids) in enumerate(operations):
                if name=='c':
                    for stage in stages:
                        stage.execute('release',{'session':'a'})
                        stage.execute('open',{'session':'c'})
                x=None
                for stage in stages:
                    request={'session':name,'step':step,'past':past,'input_ids':ids,'activation':x,'last_tokens_only':1}
                    header,payload=pack(request)
                    x=stage.execute('prepare',unpack(header,payload))
                actual=x.cpu()
                torch.testing.assert_close(actual,expected[index],rtol=0,atol=0,msg=f'{name} step{step} past{past}')
                for stage in stages:
                    committed=stage.execute('commit',{'session':name,'step':step})
                    assert committed=={'length':past+ids.shape[1],'step':step+1}
            print(json.dumps({'test':'cached-stage-native-fixed-pages','passed':len(operations),
                              'sessions':2,'page_boundary':True,'slot_reuse':True,'bitwise_equal':True}),flush=True)
        finally:
            for stage in stages:stage.model.unload()

if __name__=='__main__':main()
