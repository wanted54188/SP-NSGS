"""One explicit protocol, unchanged solver, immutable input records."""
from pathlib import Path
import csv, json
import numpy as np
import torch
from .registry import *
from .scoring import score,mean_rows
from .evolution import _rollout,_assimilated_rollout,_SparseVorticityNudger
from .metrics import pearson,rel2,rms,tail,transfer_spectrum,spectral_l1

def load_truth(record):
    with np.load(checked(record['physical']),allow_pickle=False) as z:
        times=z['time_s'].copy();ref={k:z[k].copy() for k in ('u','v','omega','q_sgs_x','q_sgs_y','div_q_sgs','pi_sgs','K','Z','E','T_sgs')}
    with np.load(checked(record['diagnostic']),allow_pickle=False) as z:matched=z['div_q_solver_matched'].copy()
    return times,ref,matched
def preflight(model,seed,regime,ic,mode,output):
    if mode not in MODES:raise ValueError('Explicit evaluation mode required')
    if model=='clark' and seed is not None:raise ValueError('Clark is deterministic; omit --seed')
    if model!='clark' and seed not in (2026,2027,2028):raise ValueError('Use a formal seed/calibration split')
    if mode=='observation_assisted' and (regime!='baseline' or model not in VARIANTS):
        raise ValueError('Assisted protocol is defined for baseline neural models only')
    dataset=dataset_record(regime,ic,mode=='observation_assisted')
    frozen=model_record(model,seed) if model in VARIANTS else classical_record(model,seed)
    dest=new_output(output)
    return dataset,frozen,dest
def augment(rows,states,times,solver,ref,matched):
    # Shared core scoring plus the formal amplitude/source diagnostics.
    for i,(row,omega) in enumerate(zip(rows,states)):
        out=solver.rhs(omega,float(times[i]));get=lambda k:out[k].detach().cpu().numpy().astype(np.float64)
        qx,qy,div,pi=(get(k) for k in ('qx','qy','div_q','pi'));q=np.stack((qx,qy));truth=np.stack((ref['q_sgs_x'][i],ref['q_sgs_y'][i]))
        row.update(q_relative_l2=rel2(q,truth),q_Pearson=pearson(q,truth),q_RMS_ratio=rms(q)/max(rms(truth),1e-30),
                   div_Pearson_physical=pearson(div,ref['div_q_sgs'][i]),div_RMS_ratio=rms(div)/max(rms(ref['div_q_sgs'][i]),1e-30),
                   Pi_relative_l2=rel2(pi,ref['pi_sgs'][i]),Pi_bias=float(np.mean(pi-ref['pi_sgs'][i])),
                   Pi_model_mean=float(pi.mean()),Pi_reference_mean=float(ref['pi_sgs'][i].mean()),
                   SGS_transfer_spectrum_L1_matched_k1_33=spectral_l1(transfer_spectrum(omega.detach().cpu().numpy().astype(np.float64),div),transfer_spectrum(ref['omega'][i],matched[i])),
                   **tail(pi,'model'),**tail(ref['pi_sgs'][i],'reference'))
        for key,val in [('qx',qx),('qy',qy)]:
            target=ref['q_sgs_x' if key=='qx' else 'q_sgs_y'][i]
            row.update({f'{key}_relative_l2':rel2(val,target),f'{key}_Pearson':pearson(val,target),f'{key}_RMS_ratio':rms(val)/max(rms(target),1e-30)})
    return rows
def evaluate(model,seed,regime,ic,mode,device,output,solver_factory=None):
    record,frozen,dest=preflight(model,seed,regime,ic,mode,output)
    solver,_,mapping=(make_solver if solver_factory is None else solver_factory)(model,seed,regime,device)
    failure=None;reason=None
    with torch.no_grad():
        if mode=='free_running_continuation':
            # Before trajectory completion, only this one-frame NPY is opened.
            initial=np.load(checked(record['initial']),allow_pickle=False)
            times=np.asarray(record['times_s'],dtype=np.float64)
            if times.shape!=(50,) or not np.allclose(times,np.arange(1,51)/10,rtol=0,atol=1e-8):raise ValueError('Frozen continuation schedule mismatch')
            omega=solver.project(torch.as_tensor(initial,device=device,dtype=torch.float32));states=[omega]
            # Formal baseline uses float32 timestamp differences; cross-regime uses
            # float64 schedule as in its frozen evaluator. Preserve both policies.
            tt=torch.as_tensor(times,device=device,dtype=torch.float32)
            for i in range(1,len(times)):
                start,dt=(float(tt[i-1].item()),float((tt[i]-tt[i-1]).item())) if regime=='baseline' else (float(times[i-1]),float(times[i]-times[i-1]))
                try:
                    omega=solver.advance(omega,start,dt)
                    if not torch.isfinite(omega).all():raise FloatingPointError('nonfinite omega')
                    states.append(omega)
                except (FloatingPointError,RuntimeError,OverflowError) as e:
                    failure=float(times[i]);reason=str(e);break
            truth_times,ref,matched=load_truth(record)
            if not np.array_equal(ref['omega'][0],initial) or not np.array_equal(truth_times,times):raise ValueError('Post-rollout reference mismatch')
        else:
            times,ref,matched=load_truth(record)
            if mode=='apriori_reference_state':states=[torch.as_tensor(w,device=device,dtype=torch.float32) for w in ref['omega']]
            else:
                split=read_json(ROOT/f'data/splits/seed_{seed}.json');indices=torch.as_tensor(split['train_flow_indices_fixed'],device=device,dtype=torch.long)
                if len(indices)!=400:raise ValueError('Frozen 400-sensor layout required')
                obs={k:torch.as_tensor(ref[k],device=device,dtype=torch.float32) for k in ('omega','u','v')}
                tt=torch.as_tensor(times,device=device,dtype=torch.float32)
                states=_assimilated_rollout(solver,obs['omega'][0],tt,obs,_SparseVorticityNudger(solver,indices))
        rows=augment(score(solver,states,times,ref,matched,mode,regime,ic),states,times,solver,ref,matched)
    summary=mean_rows(rows)
    summary.update(model=model,seed=seed,ic=ic,regime=regime,mode=mode,number_of_scored_frames=len(rows),failure_nonfinite_time_s=failure,failure_reason=reason)
    if mode=='observation_assisted':
        summary['evaluation_name']='extended_time_sparse_observation_assisted_reconstruction'
        summary['windows']={name:mean_rows([r for r in rows if lo-1e-8<=r['time_s']<=hi+1e-8]) for name,(lo,hi) in {'A_0p1_5':(.1,5.),'B_5_10':(5.,10.),'C_0p1_10':(.1,10.)}.items()}
        summary['window_inclusion']='A and B share the t=5.0 boundary; frames are not independent replicates'
    dest.mkdir(parents=True)
    with (dest/'timewise.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    provenance=dict(mode=mode,model=frozen,reference=record,physical_runtime_mapping=mapping,initialization_time_s=.1,
                    future_sparse_observations_used=mode=='observation_assisted',future_truth_access_during_free_rollout=False,
                    training_exposure='unseen_trajectory' if ic=='ic2' or regime!='baseline' else 'same training trajectory',
                    recalibration_or_training_performed=False,scoring_source_sha256=sha256(Path(__file__)),
                    timewise_sha256=sha256(dest/'timewise.csv'),summary_sha256=sha256(dest/'summary.json'))
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return dest
