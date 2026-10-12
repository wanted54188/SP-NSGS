"""Frozen-score tabulation; uncertainty unit is seed/calibration replicate."""
import json,shutil
import pandas as pd
from .registry import ROOT,new_output,sha256
def export(output):
    dest=new_output(output);dest.mkdir(parents=True)
    f=pd.read_csv(ROOT/'data/figure_inputs/factorial.csv')
    f.to_csv(dest/'factorial_per_seed.csv',index=False)
    f.groupby(['ic','mode','variant','metric']).value.agg(['mean','std','count']).reset_index().rename(columns={'std':'sample_SD','count':'n_seeds'}).to_csv(dest/'factorial_mean_sample_SD.csv',index=False)
    c=pd.read_csv(ROOT/'data/tables/cross_regime_classical_neural_data.csv');c.to_csv(dest/'cross_regime_per_replicate.csv',index=False)
    b=c[c.regime=='baseline'][['model','replicate','Pi_Pearson_apriori','omega_mean_free_running']]
    c=c.merge(b,on=['model','replicate'],suffixes=('','_baseline'),validate='many_to_one')
    for key in ('Pi_Pearson_apriori','omega_mean_free_running'):c[key+'_difference']=c[key]-c[key+'_baseline']
    c.to_csv(dest/'cross_regime_paired_differences.csv',index=False)
    c.groupby(['model','regime','replicate_type'])[['Pi_Pearson_apriori','omega_mean_free_running','Pi_Pearson_apriori_difference','omega_mean_free_running_difference']].agg(['mean','std','count']).to_csv(dest/'cross_regime_mean_sample_SD.csv')
    # A deterministic row has pandas SD=NaN, not an artificial zero SD.
    for p in (ROOT/'data/tables').glob('*.tex'):shutil.copyfile(p,dest/p.name)
    report={p.name:sha256(p) for p in dest.iterdir()}
    (dest/'provenance.json').write_text(json.dumps(dict(plotting_tabulation_only=True,source_data_changed=False,files=report),indent=2)+'\n',encoding='utf-8')
    return dest
