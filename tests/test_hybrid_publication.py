"""Single-frame endpoint/integration-path checks; no full evaluation."""
import inspect
import numpy as np
import torch
from spnsgs.registry import ROOT,make_solver,read_json,checked
from spnsgs.hybrid import HybridSolver
from spnsgs.publication_evidence import verify
from spnsgs.metrics import rel2,pearson
torch.set_num_threads(1)
def test_endpoints_and_stage_mixing():
    base,_,_=make_solver('A0B1',2026,'baseline','cpu')
    clark,_,_=make_solver('clark',None,'baseline','cpu')
    rec=read_json(ROOT/'manifests/datasets.json')['baseline/ic1']
    with np.load(checked(rec['physical'])) as z:w=torch.tensor(z['omega'][0],dtype=torch.float32)
    with torch.no_grad():
        a=base.rhs(w,.1);c=clark.rhs(w,.1)
        for alpha,expected in [(0.,a),(1.,c)]:
            h=HybridSolver(base,alpha).rhs(w,.1)
            for key in ['qx','qy','div_q','pi','rhs']:assert torch.equal(h[key],expected[key]),(alpha,key)
        hybrid=HybridSolver(base,.5);h=hybrid.rhs(w,.1)
        for key in ['qx','qy']:assert torch.equal(h[key],.5*a[key]+.5*c[key])
        official=__import__('pandas').read_csv(ROOT/'results/frozen/hybrid/ic1/apriori/seed_2026/timewise.csv').iloc[0]
        with np.load(checked(rec['physical'])) as z:
            assert abs(rel2(h['div_q'].numpy().astype(float),z['div_q_sgs'][0])-official.div_relative_l2_physical)<5e-4
            assert abs(pearson(h['pi'].numpy().astype(float),z['pi_sgs'][0])-official.Pi_Pearson)<5e-4
    assert 'rhs' not in HybridSolver.__dict__ and 'advance' not in HybridSolver.__dict__
    source=inspect.getsource(type(base).rhs);assert 'self.closure_fields(state)' in source
def test_published_archive_checks():assert verify()['status']=='PASS'
