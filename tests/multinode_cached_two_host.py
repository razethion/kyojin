"""Two-host cached greedy steps against a native single-host cached oracle."""
import argparse,json,os,time
from pathlib import Path
os.environ['EXL3_KVWIN']='0'
import torch
from safetensors.torch import load_file,save_file
from tools.multinode.stage import load_stage,source_fingerprint
from tools.multinode.worker import CachedPipeline,StageClient
from urllib.request import urlopen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('role',choices=('reference','coordinator'))
    ap.add_argument('--directory',required=True)
    ap.add_argument('--token-file')
    args=ap.parse_args()
    root=Path(args.directory)
    torch.set_num_threads(2)
    if args.role=='coordinator':
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        assert available>=6*1024**3,available
    torch.cuda.set_per_process_memory_fraction((512*1024**2)/torch.cuda.get_device_properties(0).total_memory,0)
    if args.role=='reference':
        from exllamav3 import Config,Model,Cache
        model=Model.from_config(Config.from_directory(str(root)))
        cache=Cache(model,max_num_tokens=1024)
        model.load(device='cuda:0',progressbar=False)
        lengths={'a':0,'b':0,'c':0};steps=dict.fromkeys(lengths,0)
        tokens={};tensors={};schedule=[]
        def run(name,ids):
            index=len(schedule);slot=1 if name=='b' else 0
            params={'attn_mode':'flash_attn','cache':cache,'last_tokens_only':1,
                    'block_table':torch.arange(slot*2,(slot+1)*2,dtype=torch.int32).view(1,2),
                    'cache_seqlens':torch.tensor([lengths[name]],dtype=torch.int32)}
            with torch.inference_mode():output=model.forward(ids,params).cpu()
            assert torch.isfinite(output).all()
            tensors[f'ids_{index}']=ids.clone()
            tensors[f'logits_{index}']=output
            schedule.append({'session':name,'step':steps[name],'past':lengths[name],'rows':ids.shape[1]})
            tokens[name]=output[:,-1,:].argmax(-1,keepdim=True)
            lengths[name]+=ids.shape[1];steps[name]+=1
        torch.manual_seed(399)
        run('a',torch.randint(0,256,(1,17),dtype=torch.long))
        run('b',torch.randint(0,256,(1,250),dtype=torch.long))
        for _ in range(12):
            run('a',tokens['a']);run('b',tokens['b'])
        run('c',torch.randint(0,256,(1,7),dtype=torch.long))
        for _ in range(3):run('c',tokens['c'])
        model.unload()
        save_file(tensors,str(root/'cached-oracle.bin'))
        (root/'cached-schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
        print(json.dumps({'reference_steps':len(schedule),'final_lengths':lengths,'source_id':source_fingerprint(Path.cwd())}),flush=True)
        return
    stage=load_stage(str(root),0,2,'cuda:0',str(root/'integrity.json'),max_rows=256,cache_context=512,cache_sessions=2)
    class Local:
        manifest=stage.manifest
        def call(self,op,args):return stage.execute(op,args)
        def close(self):stage.model.unload()
    remote=StageClient('10.203.0.2',9032,bytes.fromhex(Path(args.token_file).read_text().strip()),timeout=30)
    pipeline=None
    try:
        pipeline=CachedPipeline([Local(),remote]);pipeline.open('a');pipeline.open('b')
        tensors=load_file(str(root/'cached-oracle.bin'))
        schedule=json.loads((root/'cached-schedule.json').read_text())
        cases=[]
        for index,record in enumerate(schedule):
            name=record['session']
            if name=='c' and record['step']==0:
                pipeline.release('a');pipeline.open('c')
            state=pipeline.sessions.items[name]
            assert (state.length,state.step)==(record['past'],record['step'])
            start=time.perf_counter()
            output=pipeline.forward(name,tensors[f'ids_{index}'])
            expected=tensors[f'logits_{index}']
            torch.testing.assert_close(output,expected,rtol=0,atol=0,msg=f'{record}')
            assert torch.equal(output.argmax(-1),expected.argmax(-1))
            cases.append(dict(record,bitwise_equal=True,ms=(time.perf_counter()-start)*1000))
        with urlopen('http://127.0.0.1:8731/health',timeout=5) as response:health=json.loads(response.read())['status']
        assert health=='ok'
        report={'test':'two-physical-GPU-cached','cases':cases,'sessions':2,'page_boundary':True,'slot_reuse':True,
                'source_id':stage.manifest['source_id'],'model_id':stage.manifest['model_id'],
                'peak_allocated':torch.cuda.max_memory_allocated(),'primary_health':health}
        (root/'cached-two-host-result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed_steps':len(cases),'source_id':report['source_id'],'peak_allocated':report['peak_allocated'],
                          'primary_health':health,'warm_last_10_ms':[case['ms'] for case in cases[-10:]]}),flush=True)
    finally:
        if pipeline:pipeline.close()
        else:remote.close();stage.model.unload()

if __name__=='__main__':main()
