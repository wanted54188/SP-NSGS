"""The same periodic export reader and Gaussian physical-label construction."""
from pathlib import Path
import json
import numpy as np
from .config import load_config
from .data import Snapshot,discover_files,read_snapshot,inspect_reference,load_observation_snapshots
from .filters import periodic_divergence
from .metrics import _radial_spectrum,transfer_spectrum
from .registry import ROOT,REGIMES,BASE_DELTA,read_json,sha256,new_output

def scan(raw_dir,config,end=5.):
    files=[(t,p) for t,p in discover_files(Path(raw_dir),config.data.file_glob) if .1-1e-8<=t<=end+1e-8]
    expected=np.arange(1,round(end*10)+1)/10
    if len(files)!=len(expected) or not np.allclose([t for t,p in files],expected,rtol=0,atol=1e-8):raise ValueError('Missing, duplicate, or off-cadence reference frames')
    return files
def prepare(raw_dir,regime,ic,output,end=5.,execute=False):
    config=load_config(ROOT/'configs/A0B1/seed_2026.json')
    config.data.data_dir=Path(raw_dir);files=scan(raw_dir,config,end)
    first=read_snapshot(files[0][1],files[0][0],config)
    if (first.nx,first.ny)!=(100,100):raise ValueError('Unique periodic grid must be 100x100')
    delta={'filter_small':.1884955592,'filter_large':.3769911184}.get(regime,BASE_DELTA)
    if regime=='baseline' or regime.startswith('filter_'):
        old=read_json(ROOT/'manifests/datasets.json')[f'baseline/{ic}']
        hashes={r['file']:r for r in old['raw_exports']}
        for t,p in files[:50]:
            if p.name not in hashes or sha256(p)!=hashes[p.name]['sha256']:raise ValueError(f'Raw frozen export mismatch: {p.name}')
        with np.load(ROOT/old['physical']['path'],allow_pickle=False) as z:
            for name in ('u','v','omega','q_sgs_x','q_sgs_y','div_q_sgs','pi_sgs'):
                if not np.allclose(getattr(first,name).reshape(100,100),z[name][0],rtol=2e-5,atol=2e-5):raise ValueError(f'Base-width sanity check failed: {name}')
    dest=new_output(output)
    report=dict(regime=regime,ic=ic,frame_count=len(files),delta_m=delta,periodic_shape=[100,100],executed=execute,
                source_exports=[dict(file=p.name,time_s=t,sha256=sha256(p)) for t,p in files])
    if not execute:return report
    config.data.filter_width_m=delta
    names=('u','v','omega','q_sgs_x','q_sgs_y','div_q_sgs','pi_sgs')
    values={k:[] for k in (*names,'K','Z','E','T_sgs','time_s')}
    for t,p in files:
        snap=read_snapshot(p,t,config)
        if (snap.nx,snap.ny)!=(100,100):raise ValueError('Grid changed')
        frame={k:np.asarray(getattr(snap,k)).reshape(100,100) for k in names}
        if not all(np.isfinite(a).all() for a in frame.values()):raise ValueError('Nonfinite source')
        for k,a in frame.items():values[k].append(a)
        _,E=_radial_spectrum(frame['u'],frame['v']);T=transfer_spectrum(frame['omega'],frame['div_q_sgs'])
        if abs(T.sum()+frame['pi_sgs'].mean())>1e-8:raise ValueError('Physical SGS budget failed')
        values['K'].append(.5*np.mean(frame['u']**2+frame['v']**2));values['Z'].append(.5*np.mean(frame['omega']**2))
        values['E'].append(E);values['T_sgs'].append(T);values['time_s'].append(t)
    stacked={k:np.stack(v) for k,v in values.items()}
    stacked['x']=snap.x.reshape(100,100)[0];stacked['y']=snap.y.reshape(100,100)[:,0]
    kx=np.fft.fftfreq(100,d=1/100);ky=kx
    mask=(np.abs(kx)[None,:]<=33)&(np.abs(ky)[:,None]<=33)
    matched=np.stack([periodic_divergence(np.fft.ifft2(np.fft.fft2(qx)*mask).real,np.fft.ifft2(np.fft.fft2(qy)*mask).real,snap.dx,snap.dy) for qx,qy in zip(stacked['q_sgs_x'],stacked['q_sgs_y'])])
    if regime=='forced_strong':
        curl=np.broadcast_to(-4*.01*np.exp(-delta**2*16/24)*np.cos(4*stacked['y'])[:,None],(100,100)).copy()
        stacked['filtered_curl_f']=curl
        report['force']=dict(source_kind='per_volume',source_N_m3=10.,rho_kg_m3=1000.,acceleration_m_s2=.01,wave_number_rad_m=4.,included_in_SGS_labels=False)
    report['nu_m2_s']={'half_Re':2e-6,'double_Re':5e-7}.get(regime,1e-6)
    dest.mkdir(parents=True);np.savez_compressed(dest/'physical.npz',**stacked);np.savez_compressed(dest/'diagnostic.npz',div_q_solver_matched=matched)
    np.save(dest/'initial_state.npy',stacked['omega'][0],allow_pickle=False)
    report['output_hashes']={p.name:sha256(p) for p in dest.iterdir()}
    (dest/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def load_training_snapshots(config):
    path=Path(config.data.data_dir)
    if path.is_dir():return load_observation_snapshots(config)
    # The trainer consumes exactly these seven frozen fields. This adapter
    # avoids re-filtering or new normalization; numerical training bodies stay unchanged.
    rec=read_json(ROOT/'manifests/datasets.json')['baseline/ic1']
    if sha256(path)!=rec['physical']['sha256']:raise ValueError('Training must use frozen IC-1 archive')
    with np.load(path,allow_pickle=False) as z:
        xx,yy=np.meshgrid(z['x'],z['y']);snapshots=[]
        for i,t in enumerate(z['time_s']):
            values={k:z[k][i].ravel().copy() for k in ('u','v','omega','q_sgs_x','q_sgs_y','div_q_sgs','pi_sgs')}
            snapshots.append(Snapshot(time_s=float(t),source=path,x=xx.ravel(),y=yy.ravel(),nx=100,ny=100,dx=2*np.pi/100,dy=2*np.pi/100,**values))
    return snapshots
def inspect_training_reference(config):
    path=Path(config.data.data_dir)
    if path.is_dir():return inspect_reference(config)
    snaps=load_training_snapshots(config)
    return dict(source='frozen processed IC-1 archive',sha256=sha256(path),snapshot_count=len(snaps),periodic_shape=[100,100],no_refiltering=True)
