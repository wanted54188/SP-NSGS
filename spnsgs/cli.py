"""Explicit execution gates for training/evaluation and stable plot-only entry."""
from pathlib import Path
import argparse,json
from .registry import ROOT,REGIMES,VARIANTS,MODES,read_json,sha256,new_output

def parser():
    p=argparse.ArgumentParser(prog='python -m spnsgs')
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('prepare-reference',help='scan first; construct physical SGS labels only with --execute')
    r.add_argument('--raw-dir',type=Path,required=True);r.add_argument('--ic',choices=('ic1','ic2'),required=True)
    r.add_argument('--regime',choices=REGIMES,default='baseline');r.add_argument('--end-time',type=float,choices=(5.,10.),default=5.);r.add_argument('--output',type=Path,required=True);r.add_argument('--execute',action='store_true')
    t=sub.add_parser('train',help='original frozen training, random initialization; preflight default')
    t.add_argument('--variant',choices=VARIANTS,required=True);t.add_argument('--seed',type=int,choices=(2026,2027,2028),required=True)
    t.add_argument('--raw-dir',type=Path,help='optional verified original raw IC-1 exports; default frozen processed archive')
    t.add_argument('--device',choices=('cpu','cuda'),default='cpu');t.add_argument('--output',type=Path,required=True);t.add_argument('--execute',action='store_true')
    e=sub.add_parser('evaluate',help='frozen model/coefficient; preflight default')
    e.add_argument('--model',choices=VARIANTS+('leith','clark','mixed'),required=True);e.add_argument('--seed',type=int,choices=(2026,2027,2028))
    e.add_argument('--ic',choices=('ic1','ic2'),required=True);e.add_argument('--regime',choices=REGIMES,default='baseline')
    e.add_argument('--mode',choices=MODES,required=True);e.add_argument('--device',choices=('cpu','cuda'),default='cpu');e.add_argument('--output',type=Path,required=True);e.add_argument('--execute',action='store_true')
    c=sub.add_parser('calibrate',help='reproduce original IC-1 sparse classical fit; never fit on transfer data')
    c.add_argument('--seed',type=int,choices=(2026,2027,2028),required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--execute',action='store_true')
    f=sub.add_parser('figures',help='plot-only reproduction from frozen records')
    f.add_argument('--output',type=Path,required=True);f.add_argument('--dpi',type=int,default=600);f.add_argument('--selection',choices=('all','main','supplement'),default='all')
    tab=sub.add_parser('tables',help='export frozen per-seed, mean/sample SD and paired difference tables')
    tab.add_argument('--output',type=Path,required=True)
    sub.add_parser('verify',help='hash, field, split, metric and availability checks; no neural inference')
    h=sub.add_parser('hybrid-evaluate',help='published fixed alpha=.5 only; preflight unless --execute')
    h.add_argument('--ic',choices=('ic1','ic2'),required=True);h.add_argument('--seed',type=int,choices=(2026,2027,2028),required=True)
    h.add_argument('--mode',choices=('apriori_reference_state','free_running_continuation'),required=True)
    h.add_argument('--device',choices=('cpu','cuda'),default='cpu');h.add_argument('--output',type=Path,required=True);h.add_argument('--execute',action='store_true')
    pe=sub.add_parser('paper-evidence',help='offline S14-S17 and Table S5 from shipped frozen arrays')
    pe.add_argument('--output',type=Path,required=True);pe.add_argument('--dpi',type=int,default=600)
    sub.add_parser('verify-paper-evidence',help='offline spectral and fixed-Hybrid archive validation')
    return p
def main():
    a=parser().parse_args()
    if a.command=='hybrid-evaluate':
        from .hybrid import evaluate
        print(evaluate(a.ic,a.seed,a.mode,a.device,a.output,a.execute));return
    if a.command=='paper-evidence':
        from .publication_evidence import generate
        print(generate(a.output,a.dpi));return
    if a.command=='verify-paper-evidence':
        from .publication_evidence import verify
        print(json.dumps(verify(),indent=2));return
    if a.command=='prepare-reference':
        from .reference import prepare
        print(json.dumps(prepare(a.raw_dir,a.regime,a.ic,a.output,a.end_time,a.execute),indent=2));return
    if a.command=='evaluate':
        from .evaluation import preflight,evaluate
        d,m,o=preflight(a.model,a.seed,a.regime,a.ic,a.mode,a.output)
        print(f'PREFLIGHT PASS: {a.ic}/{a.regime}/{a.mode}/{a.model}/{a.seed}; {len(d["times_s"])} frames')
        if a.execute:print(evaluate(a.model,a.seed,a.regime,a.ic,a.mode,a.device,a.output))
        else:print('No model inference. Add --execute for the full run.')
        return
    if a.command=='train':
        import torch
        from .config import load_config
        from .reference import load_training_snapshots
        from .protocols import _sensor_protocol
        from .training import train_paper
        from .registry import dataset_record
        config=load_config(ROOT/f'configs/{a.variant}/seed_{a.seed}.json');dest=new_output(a.output)
        dataset=dataset_record('baseline','ic1')
        config.data.data_dir=a.raw_dir.resolve() if a.raw_dir else ROOT/dataset['physical']['path']
        if a.raw_dir:
            from .reference import scan
            files=scan(a.raw_dir,config);expected={r['file']:r['sha256'] for r in dataset['raw_exports']}
            if any(expected.get(p.name)!=sha256(p) for t,p in files):raise ValueError('Frozen raw IC-1 hashes do not match')
        config.training.device=a.device;config.training.output_dir=dest
        snaps=load_training_snapshots(config)
        ref={k:torch.tensor(__import__('numpy').stack([getattr(s,k).reshape(100,100) for s in snaps]),dtype=torch.float32) for k in ('u','v','omega')}
        *_,protocol=_sensor_protocol(config,snaps,ref,torch.device('cpu'))
        split=read_json(ROOT/f'data/splits/seed_{a.seed}.json')
        pairs=[('train_indices','train_flow_indices_fixed'),('validation_indices','validation_flow_indices_fixed'),('closure_indices_by_time','train_sgs_indices_by_time'),('closure_weights_by_time','train_sgs_weights_by_time')]
        if any(protocol[x]!=split[y] for x,y in pairs):raise ValueError('Frozen sparse split mismatch')
        print(f'TRAIN PREFLIGHT PASS: {a.variant}/{a.seed}; 400 flow / 25 SGS / 100 held-out, random initialization')
        if not a.execute:print('No training. Add --execute.');return
        dest.mkdir(parents=True)
        # Save effective runtime paths separately; frozen input config is never changed.
        from dataclasses import asdict
        (dest/'effective_config.json').write_text(json.dumps(asdict(config),default=str,indent=2)+'\n',encoding='utf-8')
        (dest/'provenance.json').write_text(json.dumps(dict(training_dataset_sha256=dataset['physical']['sha256'],seed=a.seed,variant=a.variant,split_sha256=sha256(ROOT/f'data/splits/seed_{a.seed}.json'),initialized_from_checkpoint=False),indent=2)+'\n',encoding='utf-8')
        print(train_paper(config));return
    if a.command=='calibrate':
        from .calibration import calibrate
        print(calibrate(a.seed,a.output,a.execute));return
    if a.command=='figures':
        from .figures import reproduce
        print(reproduce(a.output,a.selection,a.dpi));return
    if a.command=='tables':
        from .tables import export
        print(export(a.output));return
    from .verification import verify
    print(json.dumps(verify(),indent=2))
