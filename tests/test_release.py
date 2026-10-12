"""Light regression tests: one frozen frame only, never a 50-frame inference."""
import ast,json
import numpy as np
import pandas as pd
import torch
import pytest
from spnsgs.registry import *
from spnsgs.filters import gaussian_filter_2d,periodic_divergence
from spnsgs.classical_baselines import classical_fluxes,fit_nonnegative_leith_scalar
from spnsgs.metrics import rel2,pearson,event_metrics,transfer_spectrum
from spnsgs.verification import verify
torch.set_num_threads(1)

def test_gaussian_and_periodic_operators():
    x=np.arange(100)*2*np.pi/100;xx,yy=np.meshgrid(x,x);a=np.sin(3*xx+4*yy)
    f,gx,gy=gaussian_filter_2d(a,2*np.pi/100,2*np.pi/100,BASE_DELTA);gain=np.exp(-BASE_DELTA**2*25/24)
    assert np.max(np.abs(f-gain*a))<1e-13
    assert np.max(np.abs(gx-3*gain*np.cos(3*xx+4*yy)))<1e-12
    assert np.max(np.abs(gy-4*gain*np.cos(3*xx+4*yy)))<1e-12
def test_flux_fit_and_leith_machine_precision():
    cfg=load_config(ROOT/'configs/A0B1/seed_2026.json');s=FrozenClassicalSolver(cfg,'leith',.0025,'cpu')
    xx,yy=torch.meshgrid(torch.arange(100)*2*np.pi/100,torch.arange(100)*2*np.pi/100,indexing='ij');state=s.state(torch.sin(3*xx+4*yy));f=classical_fluxes(state,BASE_DELTA,.0025)
    assert float(f['pi_leith'].min())>=0
    analytic=.0025*BASE_DELTA**3*(state['omega_x']**2+state['omega_y']**2)**1.5
    assert torch.allclose(f['pi_leith'],analytic,rtol=5e-7,atol=1e-9)
    shape=np.arange(24.).reshape(3,4,2);w=np.ones((3,4))/4
    assert fit_nonnegative_leith_scalar(shape,.4*shape,w)==pytest.approx(.4)
    assert fit_nonnegative_leith_scalar(shape,-shape,w)==0
    assert fit_nonnegative_leith_scalar(np.zeros_like(shape),shape,w)==0
    assert fit_nonnegative_leith_scalar(shape,.4*shape+shape,w,clark_flux=shape)==pytest.approx(.4)
def test_forcing_viscosity_and_rhs_projection():
    s,_,mapping=make_solver('clark',None,'forced_strong','cpu');zero=torch.zeros((100,100))
    y=torch.arange(100)*(2*np.pi/100)
    exact=(-.04*np.exp(-BASE_DELTA**2*16/24)*torch.cos(4*y))[:,None].expand_as(zero)
    force=s.vorticity_forcing(.1,zero)
    assert torch.allclose(force,exact,rtol=2e-6,atol=2e-8)
    assert torch.allclose(s.rhs(zero,.1)['rhs'],s.project(force),rtol=2e-6,atol=2e-8)
    for regime,nu in [('half_Re',2e-6),('double_Re',5e-7),('baseline',1e-6)]:
        s,_,m=make_solver('clark',None,regime,'cpu');assert m['nu_m2_s']==nu
    k=np.fft.fftfreq(100,d=1/100);mask=(np.abs(k)[None,:]<=33)&(np.abs(k)[:,None]<=33)
    assert np.array_equal(s.dealias.cpu().numpy().astype(bool),mask)
def test_one_frame_checkpoint_and_frozen_metrics():
    solver,record,_=make_solver('A0B1',2026,'baseline','cpu');d=dataset_record('baseline','ic1')
    with np.load(checked(d['physical']),allow_pickle=False) as z:
        w=torch.as_tensor(z['omega'][0],dtype=torch.float32)
        with torch.no_grad():out=solver.rhs(w,.1)
        pi=out['pi'].numpy().astype(np.float64);div=out['div_q'].numpy().astype(np.float64)
        official=pd.read_csv(ROOT/'results/frozen/neural/ic1/A0B1_2026_apriori/timewise.csv').iloc[0]
        assert rel2(div,z['div_q_sgs'][0])==pytest.approx(official.div_relative_l2_physical,rel=1e-4,abs=2e-6)
        assert pearson(pi,z['pi_sgs'][0])==pytest.approx(official.Pi_Pearson,abs=2e-6)
        assert event_metrics(pi,z['pi_sgs'][0])['negative_F1']==pytest.approx(official.negative_F1,abs=2e-6)
        closure=solver.closure_fields(solver.state(w))
        scale=float(solver.closure_output_scale)
        assert torch.allclose(closure['qx'],scale*(closure['qd_x']+closure['qb_x']),rtol=2e-6,atol=2e-6)
        assert torch.allclose(closure['qy'],scale*(closure['qd_y']+closure['qb_y']),rtol=2e-6,atol=2e-6)
        assert abs(transfer_spectrum(z['omega'][0],z['div_q_sgs'][0]).sum()+z['pi_sgs'][0].mean())<1e-8
def test_information_access_by_construction():
    from spnsgs.evolution import _rollout,_assimilated_rollout
    assert 'reference' not in __import__('inspect').signature(_rollout).parameters
    assert 'reference' in __import__('inspect').signature(_assimilated_rollout).parameters
    from spnsgs.evaluation import evaluate
    text=__import__('inspect').getsource(evaluate)
    assert text.index('initial=np.load')<text.index('solver.advance')<text.index('truth_times,ref,matched=load_truth')
def test_missing_full2027_is_not_substituted():
    with pytest.raises(FileNotFoundError,match='MISSING EXACT FROZEN CHECKPOINT'):model_record('full',2027)
    assert (ROOT/'results/frozen/neural/ic1/full_2027_free_running/summary.json').exists()
def test_frozen_tables_and_hashes():assert verify()['status']=='PASS'

def test_shared_training_metadata_remains_hash_checked():
    models=read_json(ROOT/'manifests/models.json')
    for rec in models.values():
        for artifact in rec.get('metadata_artifacts',{}).values():checked(artifact)
