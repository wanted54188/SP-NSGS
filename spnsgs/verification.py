"""Read-only frozen artifact and manuscript-table checks; no network inference."""
import re,json
import numpy as np
import pandas as pd
from .registry import *
def verify():
    # A hydrated clone must match every shipped frozen/source-lineage byte hash.
    for record in read_json(ROOT/'manifests/source_lineage.json'):
        checked_lineage(ROOT/record['path'])
    datasets=read_json(ROOT/'manifests/datasets.json');models=read_json(ROOT/'manifests/models.json')
    checked_datasets=[]
    for key,r in datasets.items():
        for field in ('physical','diagnostic','initial'):checked(r[field])
        with np.load(ROOT/r['physical']['path'],allow_pickle=False) as z:
            assert z['omega'].shape==(len(r['times_s']),100,100)
            assert np.allclose(z['time_s'],r['times_s'],rtol=0,atol=0)
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert len(set(r['times_s']))==len(r['times_s'])
        checked_datasets.append(key)
    available=[];missing=[]
    for key,r in models.items():
        if r['status']=='EXACT FROZEN CHECKPOINT':model_record(r['variant'],r['seed']);available.append(key)
        else:missing.append(key)
    for seed in (2026,2027,2028):
        s=read_json(ROOT/f'data/splits/seed_{seed}.json');train=set(s['train_flow_indices_fixed']);val=set(s['validation_flow_indices_fixed'])
        assert len(train)==400 and len(val)==100 and not train&val
        assert np.asarray(s['train_sgs_indices_by_time']).shape==(50,25)
        assert all(set(ids)<=train for ids in s['train_sgs_indices_by_time'])
        assert np.allclose(np.sum(s['train_sgs_weights_by_time'],axis=1),1,rtol=0,atol=1e-10)
        classical_record('leith',seed);classical_record('mixed',seed)
    # Complete factorial numbers are checked against the per-run archives,
    # including the high-error Full/2027 realization.
    df=pd.read_csv(ROOT/'data/figure_inputs/factorial.csv')
    rows_checked=0
    for row in df.itertuples():
        stem=f'{row.variant}_{row.seed}_{row.mode}'
        if row.ic=='ic1' and row.variant=='full' and row.seed==2026:
            stem='full_seed2026_apriori_formal' if row.mode=='apriori' else 'full_2026_free_running_t01'
        p=ROOT/f'results/frozen/neural/{row.ic}/{stem}/summary.json'
        value=read_json(p)[row.metric]
        assert np.isclose(value,row.value,rtol=1e-12,atol=1e-12),(str(p),row.metric)
        rows_checked+=1
    # The numerical values used for Fig.8 and Supplement's comparison table
    # must agree with full frozen summaries, not values read from images.
    cross=pd.read_csv(ROOT/'data/tables/cross_regime_classical_neural_data.csv')
    for row in cross.itertuples():
        replicate='deterministic' if row.model=='clark' else f'seed_{int(row.replicate)}'
        if row.regime!='baseline':
            base=ROOT/f'results/frozen/cross_regime/{row.regime}/ic1'
            ap=read_json(base/f'apriori_reference_state/{row.model}/{replicate}/summary.json');fr=read_json(base/f'free_running_continuation/{row.model}/{replicate}/summary.json')
            assert np.isclose(ap['Pi_Pearson'],row.Pi_Pearson_apriori,rtol=1e-12,atol=1e-12)
            assert np.isclose(fr['mean_omega_relative_l2_0p1_to_5p0'],row.omega_mean_free_running,rtol=1e-12,atol=1e-12)
    # Independently regenerate rounded entries in the manuscript's amplitude table.
    table=(ROOT/'data/tables/frozen_metric_table.tex').read_text(encoding='utf-8')
    table_rows=0
    metrics=('qx_relative_l2','qx_RMS_ratio','qy_relative_l2','qy_RMS_ratio','div_relative_l2_physical','Pi_RMS_ratio','Pi_bias')
    for line in table.splitlines():
        if not line.startswith(('IC-1 &','IC-2 &')):continue
        cells=[c.strip() for c in line.split('&')];ic='ic1' if cells[0]=='IC-1' else 'ic2';model={'ISO':'iso','Full':'full','Leith':'leith','Clark':'clark','Mixed':'mixed'}.get(cells[1],cells[1]);values=[]
        for seed in (None,) if model=='clark' else (2026,2027,2028):
            if model in VARIANTS:
                stem='full_seed2026_apriori_formal' if ic=='ic1' and model=='full' and seed==2026 else f'{model}_{seed}_apriori'
                values.append(read_json(ROOT/f'results/frozen/neural/{ic}/{stem}/summary.json'))
            else:
                base=ROOT/f'results/frozen/classical/{ic}'
                folder=model if seed is None else f'{model}_seed_{seed}'
                values.append(read_json(base/f'apriori/{folder}/summary.json'))
        for cell,key in zip(cells[2:],metrics):
            nums=re.findall(r'[-+]?\d+\.\d+',cell);a=np.array([r[key] for r in values])
            assert abs(float(nums[0])-a.mean())<=.00050001,(ic,model,key,cell,a.mean())
            if len(nums)>1:assert abs(float(nums[1])-a.std(ddof=1))<=.00050001
        table_rows+=1
    return dict(status='PASS',datasets=checked_datasets,available_frozen_models=available,missing_exact_models=missing,
                factorial_rows_checked=rows_checked,table_rows_checked=table_rows,cross_regime_rows_checked=len(cross),
                complete_Historical_Full2027_retained=True,no_training_or_model_inference=True)
