"""Relative, hash-checked identities of references and frozen models."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import torch
from .config import load_config
from .solver import SpectralVorticitySolver
from .classical_baselines import classical_fluxes

ROOT=Path(__file__).resolve().parents[1]
BASE_DELTA=.25132741228718347
MODES=('apriori_reference_state','free_running_continuation','observation_assisted')
REGIMES=('baseline','filter_small','filter_large','half_Re','double_Re','forced_strong')
VARIANTS=('iso','A1B0','A0B1','full')
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def checked(record):
    path=ROOT/record['path']
    if sha256(path)!=record['sha256']:raise ValueError(f'Frozen hash mismatch: {path}')
    return path
def checked_lineage(path):
    relative=Path(path).resolve().relative_to(ROOT).as_posix()
    entries=read_json(ROOT/'manifests/source_lineage.json')
    record=next((r for r in entries if r['path']==relative),None)
    if record is None or sha256(path)!=record['sha256']:raise ValueError(f'Frozen source lineage mismatch: {relative}')
    return Path(path)
def dataset_record(regime,ic,assisted=False):
    key=f'{"assisted" if assisted else regime}/{ic}'
    records=read_json(ROOT/'manifests/datasets.json')
    if key not in records:raise ValueError(f'No frozen reference: {key}')
    rec=records[key]
    for name in ('physical','diagnostic','initial'):checked(rec[name])
    return rec
def model_record(model,seed):
    rec=read_json(ROOT/'manifests/models.json')[f'{model}/{seed}']
    if rec['status']!='EXACT FROZEN CHECKPOINT':
        raise FileNotFoundError(f'{model}/{seed}: MISSING EXACT FROZEN CHECKPOINT; historical results remain available')
    checked(dict(path=rec['checkpoint'],sha256=rec['checkpoint_sha256']))
    checked(dict(path=rec['config'],sha256=rec['config_sha256']))
    return rec
def classical_record(model,seed):
    if model=='clark':return dict(model=model,coefficient=1.,replicate_type='deterministic')
    if seed not in (2026,2027,2028):raise ValueError('Leith/Mixed require original calibration split 2026/2027/2028')
    path=checked_lineage(ROOT/f'data/calibration/seed_{seed}.json');rec=read_json(path)
    expected=read_json(ROOT/'manifests/datasets.json')['baseline/ic1']['physical']['sha256']
    if rec['training_dataset_sha256']!=expected or rec['filter_delta_m']!=BASE_DELTA:raise ValueError('Calibration provenance mismatch')
    coefficient=rec['c_leith' if model=='leith' else 'c_mixed']
    if coefficient<0 or not math.isfinite(coefficient):raise ValueError('Invalid frozen coefficient')
    return dict(model=model,seed=seed,coefficient=coefficient,replicate_type='sparse calibration split',calibration_sha256=sha256(path),original_split_sha256=rec['split_sha256'])
def load_solver(config_path,checkpoint,device):
    config=load_config(config_path);config.training.device=device
    solver=SpectralVorticitySolver(config,100,100,2*np.pi/100,2*np.pi/100,torch.device(device),torch.float32)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('version')!=config.paper.checkpoint_format_version or payload.get('experiment_variant')!=config.paper.experiment_variant:
        raise ValueError('Checkpoint version/variant mismatch')
    solver.load_state_dict(payload['model_state']);solver.set_closure_output_scale(float(payload.get('closure_output_scale',1.)))
    solver.closure.set_backscatter_factor(1.);solver.eval()
    return solver,config
class FrozenClassicalSolver(SpectralVorticitySolver):
    def __init__(self,config,model,coefficient,device):
        super().__init__(config,100,100,2*np.pi/100,2*np.pi/100,torch.device(device),torch.float32)
        self.classical_model=model;self.classical_coefficient=coefficient
    def closure_fields(self,state):
        fields=classical_fluxes(state,self.config.data.filter_width_m,self.classical_coefficient)
        key=self.classical_model
        return {'qx':fields[f'{key}_x'],'qy':fields[f'{key}_y'],'pi':fields[f'pi_{key}']}
def apply_regime(config,solver,regime):
    if regime not in REGIMES:raise ValueError(regime)
    delta=dict(filter_small=.1884955592,filter_large=.3769911184).get(regime,BASE_DELTA)
    config.data.filter_width_m=delta;solver.closure.filter_width=delta
    config.physics.forcing_enabled=regime=='forced_strong'
    config.physics.viscosity_kind='dynamic'
    config.physics.viscosity_value=dict(half_Re=.002,double_Re=.0005).get(regime,.001)
    if config.physics.forcing_enabled:
        config.physics.momentum_source_kind='per_volume';config.physics.momentum_source_amplitude=10.
        config.physics.forcing_wave_number_rad_m=4.
        config.physics.forcing_filter_gain=math.exp(-delta**2*16/24)
    return dict(delta_m=delta,nu_m2_s=config.physics.kinematic_viscosity_m2_s,source_N_m3=10. if config.physics.forcing_enabled else 0.,density_kg_m3=config.physics.density_kg_m3)
def make_solver(model,seed,regime,device):
    if model in VARIANTS:
        rec=model_record(model,seed);solver,cfg=load_solver(ROOT/rec['config'],ROOT/rec['checkpoint'],device)
    else:
        rec=classical_record(model,seed);cfg=load_config(ROOT/'configs/A0B1/seed_2026.json')
        solver=FrozenClassicalSolver(cfg,model,rec['coefficient'],device);solver.eval()
    mapping=apply_regime(cfg,solver,regime)
    return solver,rec,mapping
def new_output(path):
    path=Path(path).resolve()
    for protected in ('data','models','results/frozen','figures/published','configs','manifests','spnsgs'):
        root=(ROOT/protected).resolve()
        if path==root or root in path.parents:raise ValueError(f'Output would overwrite frozen/input material: {path}')
    if path.exists():raise FileExistsError(f'No overwrite: {path}')
    return path
