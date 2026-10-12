"""Published fixed flux blend; inherits all RK4 stages and spectral operators."""
import math,json
from pathlib import Path
from .registry import make_solver,ROOT,sha256
from .solver import SpectralVorticitySolver
from .classical_baselines import classical_fluxes

class HybridSolver(SpectralVorticitySolver):
    def __init__(self,frozen,alpha=.5):
        if alpha not in (0.,.5,1.):
            raise ValueError('Only published .5 and endpoint verification 0/1 are supported')
        super().__init__(frozen.config,frozen.ny,frozen.nx,
                         frozen.config.domain.lx/frozen.nx,frozen.config.domain.ly/frozen.ny,
                         frozen.kx.device,frozen.kx.dtype)
        self.load_state_dict(frozen.state_dict())
        self.set_closure_output_scale(frozen.closure_output_scale)
        self.closure.set_backscatter_factor(1.)
        self.alpha=float(alpha);self.requires_grad_(False);self.eval()

    def closure_fields(self,state):
        if self.alpha==0.:return super().closure_fields(state)
        c=classical_fluxes(state,self.config.data.filter_width_m,0.)
        if self.alpha==1.:return dict(qx=c['clark_x'],qy=c['clark_y'],pi=c['pi_clark'])
        a=super().closure_fields(state)
        qx=(1-self.alpha)*a['qx']+self.alpha*c['clark_x']
        qy=(1-self.alpha)*a['qy']+self.alpha*c['clark_y']
        return dict(qx=qx,qy=qy,pi=-(qx*state['omega_x']+qy*state['omega_y']),
                    qd_x=(1-self.alpha)*a['qd_x'],qd_y=(1-self.alpha)*a['qd_y'],
                    qb_x=(1-self.alpha)*a['qb_x']+self.alpha*c['clark_x'],
                    qb_y=(1-self.alpha)*a['qb_y']+self.alpha*c['clark_y'])

def fixed_solver(seed,device):
    frozen,record,mapping=make_solver('A0B1',seed,'baseline',device)
    if frozen.closure_output_scale!=1.:raise ValueError('Unexpected formal closure scale')
    return HybridSolver(frozen,.5),record,mapping

def evaluate(ic,seed,mode,device,output,execute=False):
    from .evaluation import preflight,evaluate as baseline_evaluate
    if mode not in ('apriori_reference_state','free_running_continuation'):
        raise ValueError('No assisted or test-regime hybrid protocol is reported')
    preflight('A0B1',seed,'baseline',ic,mode,output)
    if not execute:return dict(status='PREFLIGHT PASS',alpha=.5,inference=False)
    dest=baseline_evaluate('A0B1',seed,'baseline',ic,mode,device,output,
                           solver_factory=lambda model,s,regime,dev:fixed_solver(s,dev))
    summary=json.loads((dest/'summary.json').read_text());summary.update(model='Clark-A0B1 Hybrid',alpha=.5)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    p=dest/'provenance.json';record=json.loads(p.read_text())
    record.update(closure='fixed Clark-A0B1 flux blend',alpha=.5,
                  hybrid_source_sha256=sha256(Path(__file__)),
                  summary_sha256=sha256(dest/'summary.json'),
                  each_RK4_stage_current_state=True,trajectory_interpolation=False,
                  coefficient_origin='IC1 exploratory a-priori/free-running results; v3 sparse selected 1.0',
                  historical_IC2_endpoints_viewed=True,blind_test=False)
    p.write_text(json.dumps(record,indent=2)+'\n');return str(dest)
