"""Opt-in two-GPU, two-host native stage gate; tiny random FP16 fixture only."""
import argparse,json,os,time
from pathlib import Path
os.environ['EXL3_KVWIN']='0'
import torch
from safetensors.torch import load_file,save_file
from tools.multinode.fixture import create
from tools.multinode.identity import verify
from tools.multinode.stage import load_stage,source_fingerprint
from tools.multinode.worker import Pipeline,StageClient


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('role',choices=('fixture','reference','coordinator'))
    ap.add_argument('--directory',required=True)
    ap.add_argument('--token-file')
    ap.add_argument('--host',default='10.203.0.2')
    ap.add_argument('--port',type=int,default=9032)
    args=ap.parse_args()
    root=Path(args.directory)
    if args.role=='fixture':
        create(root)
        print(json.dumps({'fixture':str(root),'identity':verify(root,root/'integrity.json')}),flush=True)
        return
    torch.set_num_threads(2)
    # Bound this tiny experiment's torch allocator on the primary host.
    torch.cuda.set_per_process_memory_fraction((512*1024**2)/torch.cuda.get_device_properties(0).total_memory,0)
    if args.role=='reference':
        from exllamav3 import Config,Model
        model=Model.from_config(Config.from_directory(str(root)))
        model.load(device='cuda:0',progressbar=False)
        outputs={}
        torch.manual_seed(1729)
        with torch.inference_mode():
            for batch in (1,2):
                for rows in (1,7,16,33):
                    ids=torch.randint(0,256,(batch,rows),dtype=torch.int64)
                    for last in sorted({1,rows}):
                        key=f'b{batch}_r{rows}_l{last}'
                        logits=model.forward(ids,{'last_tokens_only':last}).cpu()
                        assert torch.isfinite(logits).all(),key
                        outputs[key+'_ids']=ids.clone()
                        outputs[key+'_logits']=logits
        model.unload()
        save_file(outputs,str(root/'oracle.bin'))
        print(json.dumps({'reference_cases':len(outputs)//2,'source_id':source_fingerprint(Path.cwd()),'peak_allocated':torch.cuda.max_memory_allocated()}),flush=True)
        return
    token=bytes.fromhex(Path(args.token_file).read_text().strip())
    stage=load_stage(str(root),0,2,'cuda:0',str(root/'integrity.json'),max_rows=64)
    class Local:
        manifest=stage.manifest
        def forward(self,ids,activation=None,last_tokens_only=1):
            return stage.forward(dict(input_ids=ids,activation=activation,last_tokens_only=last_tokens_only))
        def close(self):stage.model.unload()
    remote=StageClient(args.host,args.port,token,timeout=30)
    pipeline=None
    try:
        pipeline=Pipeline([Local(),remote])
        oracle=load_file(str(root/'oracle.bin'))
        cases=[]
        for key in sorted(k.removesuffix('_ids') for k in oracle if k.endswith('_ids')):
            last=int(key.rsplit('_l',1)[1])
            start=time.perf_counter()
            actual=pipeline.forward(oracle[key+'_ids'],last_tokens_only=last)
            expected=oracle[key+'_logits']
            assert torch.isfinite(actual).all(),key
            torch.testing.assert_close(actual,expected,rtol=0,atol=0,msg=key)
            cases.append({'case':key,'bitwise_equal':True,'ms':(time.perf_counter()-start)*1000})
        report={'test':'two-physical-GPU-native-uncached','cases':cases,'source_id':stage.manifest['source_id'],
                'model_id':stage.manifest['model_id'],'peak_allocated':torch.cuda.max_memory_allocated(),'remote':args.host}
        (root/'two-host-result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if pipeline:pipeline.close()
        else:remote.close();stage.model.unload()

if __name__=='__main__':main()
