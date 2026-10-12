"""One original sparse IC-1 fit, never a transfer-regime calibration."""
import json
import numpy as np
import torch
from .registry import *
from .classical_baselines import fit_nonnegative_leith_scalar,classical_fluxes
def calibrate(seed,output,execute=False):
    dest=new_output(output);record=dataset_record('baseline','ic1')
    split=read_json(ROOT/f'data/splits/seed_{seed}.json');sites=np.asarray(split['train_sgs_indices_by_time']);weights=np.asarray(split['train_sgs_weights_by_time'],dtype=np.float64)
    if sites.shape!=(50,25) or weights.shape!=sites.shape:raise ValueError('Frozen 25-label layout mismatch')
    if not execute:return dict(status='PREFLIGHT PASS',seed=seed,no_fitting_performed=True)
    solver,_,_=make_solver('leith',seed,'baseline','cpu')
    selected=np.empty((50,25,2));leith=np.empty_like(selected);clark=np.empty_like(selected)
    with np.load(checked(record['physical']),allow_pickle=False) as z,torch.no_grad():
        for i,idx in enumerate(sites):
            state=solver.state(torch.as_tensor(z['omega'][i],dtype=torch.float32));formula=classical_fluxes(state,BASE_DELTA,1.)
            for component,axis in enumerate(('x','y')):
                selected[i,:,component]=z[f'q_sgs_{axis}'][i].ravel()[idx]
                leith[i,:,component]=formula[f'leith_{axis}'].numpy().ravel()[idx]
                clark[i,:,component]=formula[f'clark_{axis}'].numpy().ravel()[idx]
    cL=fit_nonnegative_leith_scalar(leith,selected,weights);cLC=fit_nonnegative_leith_scalar(leith,selected,weights,clark_flux=clark)
    old=read_json(ROOT/f'data/calibration/seed_{seed}.json')
    if not np.allclose([cL,cLC],[old['c_leith'],old['c_mixed']],rtol=2e-6,atol=1e-12):raise ValueError('Reproduced calibration differs from frozen coefficients')
    dest.mkdir(parents=True);rec=dict(seed=seed,c_leith=cL,c_mixed=cLC,training_dataset_sha256=record['physical']['sha256'],no_test_calibration=True)
    (dest/'calibration.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8');return rec
