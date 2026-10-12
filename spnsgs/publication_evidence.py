"""Publication rendering of validated frozen arrays only; no inference."""
from pathlib import Path
import sys,json,hashlib
from .registry import ROOT,sha256,new_output,checked,read_json
from .metrics import event_metrics
PRIMARY=('clark','A0B1:2026','A0B1:2027','A0B1:2028')
def verified_export(key):
    rec=read_json(ROOT/'manifests/spectral_evidence.json')[key.replace(':','_')]
    for record in rec['files'].values():checked(record)
    d=ROOT/rec['directory'];return d,read_json(d/'provenance.json')
def snapshots(model,t):
    rec=read_json(ROOT/'manifests/snapshot_evidence.json')[f'{model}/{t:g}'];p=checked(rec)
    import numpy as np
    with np.load(p,allow_pickle=False) as z:a={k:z[k].astype(np.float64) for k in z.files}
    return a,rec,[p]
def provenance(paths,**extra):
    return dict(mode='apriori_reference_state',model_inference=False,
        sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(set(paths))],**extra)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

OUT=None
DPI=600

def save(fig,name,rows,meta):
 for ext in ('pdf','svg','png'):fig.savefig(OUT/(name+'.'+ext),dpi=DPI,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)

 meta['output_sha256']={p.name:sha256(p) for p in OUT.glob(name+'.*')}
 (OUT/(name+'_provenance.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')

def generate_core(output,dpi=600):
 global OUT,DPI
 OUT=new_output(output);DPI=dpi
 OUT.mkdir(parents=True,exist_ok=False)
 loaded={};sources=[]
 for key in PRIMARY:
  d,p=verified_export(key)
  with np.load(d/'per_frame_spectra.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
  assert np.array_equal(a['shell'],np.arange(72)) and np.allclose(a['time_s'],np.arange(1,51)/10)
  assert all(a[k].shape==(50,72) for k in a if k not in ('time_s','shell'))
  loaded[key]=a;sources += [d/'per_frame_spectra.npz',d/'provenance.json',d/'per_frame_validation.csv']
 k=np.arange(72);rows=[]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12,'axes.labelsize':12,'legend.fontsize':10,'svg.fonttype':'none'})
 def curves(ax,field,power=False):
  vals={key:np.sqrt(a[field].mean(0)) if power else a[field].mean(0) for key,a in loaded.items()}
  ax.plot(k,vals['clark'],color='black',lw=2,label='Clark (deterministic)')
  seeds=np.stack([vals[key] for key in PRIMARY[1:]])
  for key,style in zip(PRIMARY[1:],(':','--','-.')):ax.plot(k,vals[key],style,color='#009E73',lw=1,alpha=.7,label='A0B1 '+key.split(':')[1])
  mean=seeds.mean(0);sd=seeds.std(0,ddof=1)
  ax.fill_between(k,mean-sd,mean+sd,color='#009E73',alpha=.16,label='A0B1 sample SD')
  ax.plot(k,mean,color='#009E73',lw=2,label='A0B1 mean')
  for key,val in vals.items():rows.extend(dict(quantity=field,run=key,shell=int(j),value=float(v)) for j,v in zip(k,val))
  ax.axvspan(.5,33.5,color='#777777',alpha=.05)
  ax.axvline(33.5,color='#777777',ls=':',lw=1)
  ax.set_xlim(0,71);ax.set_xlabel(r'Fourier shell $k$ (m$^{-1}$)');ax.grid(alpha=.15)
 fig,axes=plt.subplots(1,3,figsize=(13.2,5.2),layout='constrained')
 for i,(ax,field,title) in enumerate(zip(axes,('physical_target_error_power','operator_matched_error_power','full_fourier_constitutive_error_power'),('Physical-target','Operator-matched','Full-Fourier constitutive'))):
  curves(ax,field,True)
  rf='reference_operator_matched_source_power' if i==1 else 'reference_source_power'
  v=np.sqrt(loaded['clark'][rf].mean(0));ax.plot(k,v,color='#555555',ls=(0,(4,2)),lw=1.7,label='Reference source')
  rows.extend(dict(quantity=rf,run='reference',shell=int(j),value=float(x)) for j,x in zip(k,v))
  ax.set_yscale('symlog',linthresh=1e-4);ax.set_ylim(bottom=0)
  from matplotlib.ticker import FuncFormatter,LogFormatterMathtext
  fmt=LogFormatterMathtext(base=10)
  ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: '0 (linear)' if v==0 else fmt(v,pos)))
  ax.set_title(f'({chr(97+i)}) {title}');ax.set_ylabel(r'Source shell RMS (s$^{-2}$)')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=11)
 save(fig,'FigS14_IC1_50frame_source_spectra',rows,provenance(sources,aggregation='sqrt(mean over 50 frames of shell power) per run; mean and sample SD across three seed curves; Clark deterministic',physical_target='projected model minus unprojected physical reference (original frozen metric)',operator_matched='projected model minus projected reference',full_Fourier='unprojected constitutive model minus unprojected physical reference',complete_shells=72,frames_per_run=50))
 rows=[];fig,axes=plt.subplots(1,2,figsize=(10.8,5.0),layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
 for i,ax in enumerate(axes):
  curves(ax,'model_projected_T')
  v=loaded['clark']['reference_T'].mean(0);ax.plot(k,v,color='#555555',ls=(0,(4,2)),lw=1.8,label='Physical reference')
  ax.axhline(0,color='#888888',lw=.7);ax.set_ylabel(r'$T_{\rm SGS}(k)$ (s$^{-3}$)')
  ax.set_title('(a) Complete shell range' if i==0 else '(b) Detail: official metric interval')
  if i==1:ax.set_xlim(.5,33.5)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=11)
 rows.extend(dict(quantity='reference_T',run='reference',shell=int(j),value=float(x)) for j,x in zip(k,v))
 save(fig,'FigS15_IC1_50frame_signed_transfer',rows,provenance(sources,aggregation='arithmetic 50-frame co-spectrum per run; seed-level mean +/- sample SD',sign='positive T = resolved enstrophy gain; sum over ALL shells = -spatial mean of frozen Pi',metric='Mean per-frame relative L1 k=1..33, not relative L1 of mean curve',complete_shells=72,frames_per_run=50))
 # Physical snapshot values, including positive predictions on negative-reference sites.
 models=('reference','iso','clark','A0B1','full');names=('Reference','ISO 2026','Clark','A0B1 2026','Full 2026 recovery01')
 colors=('#555555','#999999','#000000','#009E73','#E69F00');times=(1.,5.)
 snap={};spaths=[]
 for t in times:
  for m in models:
   a,p,paths=snapshots(m,t);snap[t,m]=a['Pi'];spaths+=paths
 cmap=ListedColormap(('#e3e3e3','#d55e00','#0072b2','#009e73'));norm=BoundaryNorm(np.arange(-.5,4.5),4)
 plt.rcParams.update({'font.size':14,'axes.labelsize':14,'axes.titlesize':14,'legend.fontsize':13})
 fig=plt.figure(figsize=(14.6,8.9),layout='constrained');grid=fig.add_gridspec(3,1,height_ratios=[1.6,1,1])
 pdfaxes=grid[0].subgridspec(1,4);rows=[];counts=[]
 for ti,t in enumerate(times):
  ref=snap[t,'reference'];truth=ref<0
  for ci in (0,1):
   ax=fig.add_subplot(pdfaxes[0,ti*2+ci]);vals=[snap[t,m][snap[t,m]<0 if ci==0 else truth] for m in models]
   pooled=np.concatenate([v for v in vals if len(v)]);edges=np.linspace(pooled.min(),pooled.max(),81)
   for m,name,color,v in zip(models,names,colors,vals):
    if len(v)==0:
     ax.text(.97,.95,'ISO: undefined\n(no negative events)',ha='right',va='top',transform=ax.transAxes,fontsize=11,color='#777777');continue
    density,_=np.histogram(v,bins=edges,density=True)
    ax.stairs(density,edges,color=color,lw=1.5,label=name)
    rows.extend(dict(time_s=t,model=m,conditioning='own_negative' if ci==0 else 'reference_negative',left=float(l),right=float(r),density=float(vv),count=len(v)) for l,r,vv in zip(edges[:-1],edges[1:],density))
   ax.axvline(0,color='#999999',lw=.6);ax.set_yscale('log');ax.grid(alpha=.13)
   ax.set_xlabel(r'Physical $\Pi_\omega$ (s$^{-3}$)');ax.set_ylabel(r'PDF (s$^3$)')
   ax.set_title(f'({chr(97+ti*2+ci)}) t = {t:g} s\n'+('Own negative events' if ci==0 else r'Condition: reference $\Pi_\omega<0$'))
  maps=grid[ti+1].subgridspec(1,5)
  for j,(m,name) in enumerate(zip(models,names)):
   ax=fig.add_subplot(maps[0,j]);pi=snap[t,m];pred=pi<0
   if m=='reference':ax.imshow(truth,cmap=ListedColormap(('#e3e3e3','#555555')),origin='lower',interpolation='nearest',extent=(0,2*np.pi,0,2*np.pi),vmin=0,vmax=1)
   else:
    cat=np.where(truth,np.where(pred,3,1),np.where(pred,2,0));ax.imshow(cat,cmap=cmap,norm=norm,origin='lower',interpolation='nearest',extent=(0,2*np.pi,0,2*np.pi))
    ev=event_metrics(pi,ref);counts.append(dict(time_s=t,model=m,**ev,TN=int((cat==0).sum()),FN=int((cat==1).sum()),FP=int((cat==2).sum()),TP=int((cat==3).sum())))
   ax.set_title(f't = {t:g} s\n'+name,fontsize=13);ax.set_xticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$']);ax.set_yticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$']);ax.tick_params(labelsize=11)
   ax.set_xlabel('x (m)',fontsize=11)
   if j==0:ax.set_ylabel('y (m)',fontsize=11)
 from matplotlib.patches import Patch
 handles=[plt.Line2D([0],[0],color=c,lw=2,label=n) for c,n in zip(colors,names)]+[Patch(facecolor=c,label=n) for c,n in zip(cmap.colors,('TN','FN','FP','TP'))]
 fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=12)
 save(fig,'FigS16_physical_negative_transfer_diagnostics',rows,provenance(spaths,snapshots=[1.,5.],event='native physical Pi < 0; no centering or normalization',own_negative_ISO='undefined (empty conditioning set)',Full_2026='recorded recovery01; no Full/2027 substitution',PDF='80 physical bins covering full observed range for each time and conditioning; density, not probability mass'))
 pd.DataFrame(counts).to_csv(OUT/'FigS16_snapshot_event_counts.csv',index=False)
 print('Publication rendering complete; frozen inputs not modified:',OUT)


def hybrid_figure():
    d=pd.read_csv(checked(read_json(ROOT/'manifests/publication_evidence.json')['data/figure_inputs/FigS17_figure_data.csv']))
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':12})
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.9),layout='constrained')
    colors={'A0B1':'#009e73','Hybrid':'#0072B2','Clark':'#202020'}
    for row,key in enumerate(['omega_relative_l2','Z_relative_difference']):
        ymax=float(d[key].max())*1.06
        for col,ic in enumerate(['ic1','ic2']):
            ax=axes[row,col]
            for model in colors:
                seeds=[2026] if model=='Clark' else [2026,2027,2028]
                rows=[d[(d.ic==ic)&(d.model==model)&(d.seed==s)].sort_values('time_s') for s in seeds]
                ys=np.stack([x[key].to_numpy() for x in rows]);t=rows[0].time_s
                if model!='Clark':
                    for y in ys:ax.plot(t,y,color=colors[model],lw=.8,alpha=.4)
                    ax.fill_between(t,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),color=colors[model],alpha=.15)
                ax.plot(t,ys.mean(0),color=colors[model],lw=1.8,label='Hybrid (α=0.5)' if model=='Hybrid' else model)
            ax.set(title=f'({chr(97+row*2+col)}) '+ic.upper().replace('IC','IC-'),xlabel='t (s)',
                ylabel='Vorticity relative L2' if row==0 else 'Relative enstrophy discrepancy',xlim=(.1,5),ylim=(0,ymax))
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            inset=ax.inset_axes([.105,.54,.38,.38]);neural=d[(d.ic==ic)&(d.model!='Clark')]
            for model in ['A0B1','Hybrid']:
                rows=[neural[(neural.model==model)&(neural.seed==s)].sort_values('time_s') for s in [2026,2027,2028]]
                ys=np.stack([x[key].to_numpy() for x in rows]);t=rows[0].time_s
                for y in ys:inset.plot(t,y,color=colors[model],lw=.65,alpha=.4)
                inset.plot(t,ys.mean(0),color=colors[model],lw=1.2)
            inset.set(xlim=(.1,5),ylim=(0,float(neural[key].max())*1.06),title='Neural detail')
            inset.title.set_fontsize(10);inset.tick_params(labelsize=9)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=11)
    return fig

def generate(output,dpi=600):
    generate_core(output,dpi)
    name='FigS17_exploratory_hybrid_free_running';fig=hybrid_figure()
    for ext in ['pdf','png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=dpi)
    plt.close(fig)
    import shutil
    shutil.copy2(ROOT/'data/tables/hybrid_benchmark_table.tex',OUT/'TableS5.tex')
    shutil.copy2(ROOT/'data/tables/HYBRID_FINAL_METRICS.csv',OUT/'TableS5_metrics.csv')
    shutil.copy2(ROOT/'data/tables/HYBRID_PAIRED_DIFFERENCES.csv',OUT/'TableS5_paired_differences.csv')
    exact=OUT/'published_exact';exact.mkdir()
    for stem in ['FigS14_IC1_50frame_source_spectra','FigS15_IC1_50frame_signed_transfer','FigS16_physical_negative_transfer_diagnostics',name]:
        shutil.copy2(ROOT/'figures/published'/(stem+'.pdf'),exact/(stem+'.pdf'))
    return str(OUT)

def verify():
    spectral=read_json(ROOT/'manifests/spectral_evidence.json');rows_checked=0
    reference_record=read_json(ROOT/'manifests/datasets.json')['baseline/ic1']
    with np.load(checked(reference_record['physical']),allow_pickle=False) as reference_archive:
        reference_T=reference_archive['T_sgs'].copy()
        reference_mean_Pi=reference_archive['pi_sgs'].mean(axis=(1,2))
    for run,record in spectral.items():
        for rec in record['files'].values():checked(rec)
        folder=ROOT/record['directory'];v=pd.read_csv(folder/'per_frame_validation.csv')
        with np.load(folder/'per_frame_spectra.npz',allow_pickle=False) as z:
            assert z['physical_target_error_power'].shape==(50,72)
            assert np.allclose(z['reference_T'],reference_T,rtol=1e-10,atol=1e-10)
            assert np.max(np.abs(z['reference_T'].sum(1)+reference_mean_Pi))<1e-8
            for kind,target,col in [('physical_target','reference_source_power','div_relative_l2_physical'),
                                    ('operator_matched','reference_operator_matched_source_power','div_relative_l2_operator_matched')]:
                e=np.sqrt(z[kind+'_error_power'].sum(1)/z[target].sum(1))
                assert np.allclose(e,v[col],rtol=1e-12,atol=1e-12)
            l1=np.abs(z['model_projected_T'][:,1:34]-z['reference_T'][:,1:34]).sum(1)/np.abs(z['reference_T'][:,1:34]).sum(1)
            assert np.allclose(l1,v.SGS_transfer_spectrum_L1_physical_k1_33,rtol=1e-12,atol=1e-12)
            scale=np.maximum(v.Pi_model_RMS.to_numpy(),1.)
            assert np.max(np.abs(z['model_projected_T'].sum(1)+v.SGS_budget_mean_Pi)/scale)<3e-6
        # Validate against the actual original frozen result, not only export self-check.
        official=ROOT/('results/frozen/classical/ic1/apriori/clark' if run=='clark' else f'results/frozen/neural/ic1/{run}_apriori')/'timewise.csv'
        baseline=pd.read_csv(official)
        for col in ['div_relative_l2_physical','SGS_transfer_spectrum_L1_physical_k1_33','Pi_Pearson','negative_F1']:
            assert np.allclose(v[col],baseline[col],rtol=5e-4,atol=5e-4),(run,col)
        rows_checked+=len(v)
    for rec in read_json(ROOT/'manifests/snapshot_evidence.json').values():checked(rec)
    meta=read_json(ROOT/'manifests/hybrid.json');table=pd.read_csv(ROOT/'data/tables/HYBRID_FINAL_METRICS.csv')
    from datetime import datetime
    lock=datetime.fromisoformat(meta['lock_created_utc'])
    for key,rec in meta['runs'].items():
        for asset in rec['files'].values():checked(asset)
        df=pd.read_csv(ROOT/rec['directory']/'timewise.csv');ic,mode,seed=key.split('/')
        row=table[(table.ic==ic)&(table['mode']==mode)&(table.model=='Hybrid')&(table.statistic=='per_seed')&(table.seed==int(seed))].iloc[0]
        assert len(df)==50
        for col in ['q_vector_relative_l2','div_relative_l2_physical','omega_relative_l2','Z_relative_difference','Pi_Pearson','negative_F1']:
            assert np.isclose(df[col].mean(),row[col],rtol=1e-12,atol=1e-12),(key,col)
        if ic=='ic2':assert datetime.fromisoformat(rec['created_utc'])>lock
    # Verify all baseline endpoint summaries and per-seed plotting inputs.
    plot=pd.read_csv(ROOT/'data/figure_inputs/FigS17_figure_data.csv')
    for row in table[table.statistic.isin(['per_seed','deterministic'])].itertuples():
        df=pd.read_csv(ROOT/row.source_directory/'timewise.csv')
        for col in ['div_relative_l2_physical','omega_relative_l2','Z_relative_difference','Pi_Pearson']:
            assert np.isclose(df[col].mean(),getattr(row,col),rtol=5e-4,atol=5e-4),(row.model,col)
        if row.mode=='free_running':
            p=plot[(plot.ic==row.ic)&(plot.model==row.model)&(plot.seed==int(row.seed))].sort_values('time_s')
            for col in ['time_s','omega_relative_l2','Z_relative_difference']:
                assert np.allclose(p[col],df[col],rtol=1e-12,atol=1e-12),(row.model,col)
    return dict(status='PASS',spectral_model_states=rows_checked,hybrid_runs=12,alpha=.5,
        timestamp_order_verified=True,sign='positive T = resolved enstrophy gain; complete sum = -mean(Pi)',
        no_inference=True,Full_2027='MISSING EXACT FROZEN CHECKPOINT')
