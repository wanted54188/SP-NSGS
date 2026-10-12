"""Plot-only regeneration of manuscript-selected evidence from frozen CSV/NPZ.

Published assets are also recovered byte-for-byte in published_exact/. Numeric
redraws use explicit layouts below; they are not claimed to be pixel-identical.
No model inference, calibration, rollout, or training is performed here.
"""
from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from .registry import ROOT,read_json,sha256,new_output,checked_lineage

COLORS={'iso':'#777777','A1B0':'#0072B2','A0B1':'#009E73','full':'#D55E00','leith':'#7440a3','clark':'#111111','mixed':'#d657a5','reference':'#111111'}
LABELS={'iso':'ISO','full':'Full','leith':'Leith','clark':'Clark','mixed':'Mixed','reference':'Reference','A0B1':'A0B1','A1B0':'A1B0'}
REGIMES=('baseline','filter_small','filter_large','half_Re','double_Re','forced_strong')
BLUE,GREEN,ORANGE='#1769a6','#12805d','#b7540a'
TEXT,MUTED,LINE='#20252b','#505b64','#d7dfe5'
RC=['#555555','#2878b5','#d95f02','#1b9e77','#7570b3','#c51b7d']
def csv(name):return pd.read_csv(checked_lineage(ROOT/'data/figure_inputs'/f'{name}.csv'))
def grid(data,value='value',x='ix',y='iy'):
    return data.pivot(index=y,columns=x,values=value).sort_index().sort_index(axis=1).to_numpy()
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
def dots(ax,data,xcol,models,value='value',group='ic'):
    groups=list(data[group].dropna().unique()) if group in data else ['all']
    for x,m in enumerate(models):
        for j,g in enumerate(groups):
            subset=data[(data[xcol]==m)&(data[group]==g)] if group in data else data[data[xcol]==m]
            vals=subset[value].dropna().to_numpy();pos=x+(j-(len(groups)-1)/2)*.2;marker=('o','D','s')[j%3]
            if not len(vals):continue
            ax.scatter(pos+np.linspace(-.035,.035,len(vals)),vals,marker=marker,facecolors='none',edgecolors=COLORS.get(m,'#2878b5'),s=25)
            ax.errorbar(pos,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else None,fmt=marker,color=COLORS.get(m,'#2878b5'),capsize=3,ms=6)
    ax.set_xticks(range(len(models)),[LABELS.get(m,m) for m in models]);clean(ax)
def maps(fig,axes,frames,titles,bound=None,normalized=False):
    if bound is None:bound=max(np.max(np.abs(x)) for x in frames)
    for ax,a,title in zip(axes,frames,titles):
        im=ax.imshow(a,origin='lower',extent=(0,2*np.pi,0,2*np.pi),interpolation='nearest',cmap='RdBu_r',vmin=-bound,vmax=bound)
        ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.colorbar(im,ax=list(axes),shrink=.8,extend='both' if normalized else 'neither')
def curves(ax,df,group='model'):
    replicate='replicate' if 'replicate' in df else 'seed'
    for m,g in df.groupby(group,sort=False):
        p=g.pivot(index='time_s',columns=replicate,values='omega_relative_l2')
        for col in p:ax.plot(p.index,p[col],color=COLORS.get(m,'#444'),lw=.65,alpha=.35)
        ax.plot(p.index,p.mean(axis=1),color=COLORS.get(m,'#444'),label=LABELS.get(m,m),lw=1.6)
    ax.set_xlabel('Time (s)');ax.set_ylabel(r'$\omega$ relative $L^2$');ax.set_ylim(bottom=0);clean(ax)
def rounded(ax, x, y, w, h, face, edge, radius=.012, lw=1):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw,
                           transform=ax.transAxes)
    ax.add_patch(patch)

def text(ax, x, y, value, size=9.2, color=TEXT, weight="normal", va="center", ha="left"):
    ax.text(x, y, value, transform=ax.transAxes, fontsize=size, color=color,
            weight=weight, va=va, ha=ha, linespacing=1.28)


def method():
    fig, ax = plt.subplots(figsize=(11.7, 6.0))
    fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.025)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Two parallel branches: the diagram must not imply q_d is an input to q_b.
    text(ax, .035, .96, "(a) Constrained SGS-flux decomposition", 12.0, weight="bold")
    text(ax, .965, .96, r"$q_{\mathrm{SGS}}=q_d+q_b$", 13.2, weight="bold", ha="right")
    rounded(ax, .035, .715, .45, .18, "#f2f7fb", BLUE, lw=1.5)
    rounded(ax, .515, .715, .45, .18, "#fff6ef", ORANGE, lw=1.5)
    text(ax, .055, .857, "Dissipative component", 10.2, BLUE, "bold")
    text(ax, .055, .805, r"$q_d=-\nu_d A_\theta\nabla\bar\omega,\quad A_\theta\succ0$", 11.7)
    text(ax, .055, .751, r"$\Pi_d=-q_d\cdot\nabla\bar\omega\geq0$ pointwise", 9.3, BLUE)
    text(ax, .535, .857, "Bounded signed correction", 10.2, ORANGE, "bold")
    text(ax, .535, .805, r"$q_b$: sign-indefinite local correction", 10.6)
    text(ax, .535, .751, "Allows local reverse SGS enstrophy transfer", 9.3, ORANGE)
    text(ax, .035, .678,
         "The pointwise guarantee belongs to the dissipative component; it is not a global stability theorem for the complete closure.",
         8.8, MUTED)

    ax.plot([.035, .965], [.641, .641], color=LINE, lw=1, transform=ax.transAxes)
    text(ax, .035, .606, "(b–d) What information enters each evaluation mode?", 11.3, weight="bold")

    xs = (.035, .263, .505, .748)
    ws = (.217, .231, .232, .217)
    headers = ("Mode", "Reference / observation access", "State update", "Reported quantity")
    for x, head in zip(xs, headers): text(ax, x+.012, .556, head, 8.5, MUTED, "bold")

    rows = [
        (.405, BLUE, "A-priori\nreference state",
         "Reference state\nat every time", "Closure evaluated on\neach reference state", "SGS flux, source,\ntransfer"),
        (.263, GREEN, "Free-running\ncontinuation",
         "Reference once\nat t = 0.1 s", "Solver + closure;\nmodel state evolves", "Trajectory error\nover 0.1–5 s"),
        (.121, ORANGE, "Observation-assisted\nreconstruction",
         "400 fixed sites (4%);\nω, u, v every 0.1 s", "Solver + fixed nudging;\nsparse data continue", "Reconstruction\nover 0.1–10 s"),
    ]
    records = []
    for y, color, mode, access, update, report in rows:
        rounded(ax, .035, y, .93, .128, "#ffffff", LINE, lw=.9)
        rounded(ax, xs[0], y, ws[0], .128,
                {BLUE:"#f2f7fb", GREEN:"#f0f8f5", ORANGE:"#fff6ef"}[color], color, lw=1.4)
        for x in (.258, .5, .743):
            ax.plot([x, x], [y+.016, y+.112], color=LINE, lw=.8, transform=ax.transAxes)
        text(ax, xs[0]+.015, y+.064, mode, 9.5, color, "bold")
        for x, val in zip(xs[1:], (access, update, report)):
            text(ax, x+.012, y+.064, val, 9.0)
        records.append({"mode": mode.replace("\n", " "), "access": access.replace("\n", " "),
                        "state_update": update.replace("\n", " "), "reported": report.replace("\n", " ")})

    text(ax, .035, .055,
         "Free-running after t = 0.1 s: no sensors, nudging, teacher forcing, reference reset or adaptation.",
         8.8, GREEN, "bold")

    return fig

def factorial():
    d=csv('factorial');fig,axs=plt.subplots(1,3,figsize=(12.6,4.15),layout='constrained')
    for ax,key,title in zip(axs,('Pi_Pearson','negative_F1','omega_relative_l2'),('A-priori transfer Pearson','Negative-transfer F1','Mean free-running vorticity error')):
        dots(ax,d[d.metric==key],'variant',('iso','A1B0','A0B1','full'));ax.set_title(title)
    fig.suptitle('Paired factorial attribution: IC-1 circles; IC-2 diamonds; mean ± sample SD')
    return fig
def pi_fields(density=True,normalized=False):
    d=csv('pi_snapshot');models=('reference','iso','A0B1','full');frames=[grid(d[d.model==m],'normalized_Pi' if normalized else 'raw_Pi') for m in models]
    if density:
        fig=plt.figure(figsize=(12.8,7.8),layout='constrained');gs=fig.add_gridspec(2,12)
        axs=[fig.add_subplot(gs[0,3*i:3*i+3]) for i in range(4)];maps(fig,axs,frames,[LABELS[m] for m in models])
        for j,m in enumerate(models[1:]):
            ax=fig.add_subplot(gs[1,4*j:4*j+4]);g=d[d.model==m];ref=d[d.model=='reference'].sort_values(['iy','ix']).normalized_Pi.to_numpy();val=g.sort_values(['iy','ix']).normalized_Pi.to_numpy();mask=(np.abs(ref)<=3)&(np.abs(val)<=3)
            ax.hexbin(ref[mask],val[mask],gridsize=42,extent=(-3,3,-3,3),mincnt=1,norm=LogNorm(vmin=1,vmax=500),cmap='cividis');ax.plot([-3,3],[-3,3],'r--',lw=1);ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal',title=LABELS[m],xlabel='Reference normalized Pi',ylabel='Model normalized Pi')
            ax.text(.04,.95,f'50-frame mean r={g.official_50frame_Pi_Pearson.iloc[0]:.3f}\nNegative F1={g.official_50frame_negative_F1.iloc[0]:.3f}',transform=ax.transAxes,va='top',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        fig.suptitle('IC-1, t=5 s, seed 2026: raw physical Pi (top); RMS-normalized density (bottom)')
    else:
        fig,axs=plt.subplots(1,4,figsize=(12.8,3.8),layout='constrained');maps(fig,axs,frames,[LABELS[m] for m in models],bound=3 if normalized else None,normalized=normalized)
        fig.suptitle('RMS-normalized field; amplitude removed to isolate spatial-pattern fidelity.' if normalized else 'Native physical Pi')
    return fig
def density_only():
    d=csv('pi_density');fig,axs=plt.subplots(1,3,figsize=(11.8,4.25),layout='constrained')
    for ax,m in zip(axs,('iso','A0B1','full')):
        g=d[d.model==m];mask=g.included_in_fixed_window
        ax.hexbin(g.loc[mask,'reference_normalized_Pi'],g.loc[mask,'model_normalized_Pi'],gridsize=42,extent=(-3,3,-3,3),mincnt=1,norm=LogNorm(vmin=1,vmax=500),cmap='cividis');ax.plot([-3,3],[-3,3],'r--');ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal',title=LABELS[m],xlabel='Reference normalized Pi',ylabel='Model normalized Pi')
        ax.text(.04,.95,f'50-frame mean r={g.official_50frame_Pi_Pearson.iloc[0]:.3f}\nNegative F1={g.official_50frame_negative_F1.iloc[0]:.3f}',transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none'})
    fig.suptitle('Normalized density: amplitude removed; annotations use all 50 frozen frames')
    return fig
def tradeoff():
    trade=csv('tradeoff');trade['model']=trade.model.replace({'Leith':'leith','Clark':'clark','Mixed':'mixed','Full':'full'});roll=csv('classical_rollout');fig,axs=plt.subplots(2,2,figsize=(11.8,8.1),layout='constrained')
    for row,ic in enumerate(('ic1','ic2')):
        ax=axs[row,0]
        for m,g in trade[trade.ic==ic].groupby('model'):
            ax.scatter(g.x_value,g.y_value,facecolors='none',edgecolors=COLORS[m],marker='D' if m=='clark' else 'o')
            ax.errorbar(g.x_value.mean(),g.y_value.mean(),xerr=g.x_value.std(ddof=1) if len(g)>1 else None,yerr=g.y_value.std(ddof=1) if len(g)>1 else None,fmt='D' if m=='clark' else 'o',color=COLORS[m],label=LABELS[m],capsize=3)
        ax.set(title=ic.upper(),xlabel='A-priori div(q) relative L2',ylabel='Mean free-running vorticity relative L2');clean(ax)
        curves(axs[row,1],roll[roll.ic==ic]);axs[row,1].set_ylim(0,7.6)
    axs[0,0].legend(frameon=False);return fig
def rollout(classical=True,assisted=False):
    name='assisted_rollout' if assisted else 'classical_rollout' if classical else 'neural_rollout'
    d=csv(name);fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,ic in zip(axs,('ic1','ic2')):
        curves(ax,d[d.ic==ic],group='model' if classical and not assisted else 'variant');ax.set_title(ic.upper());ax.legend(frameon=False,ncol=2)
        if classical and not assisted:ax.set_ylim(0,7.6)
        if assisted:ax.axvline(5,color='#555',ls='--',lw=.8)
    fig.suptitle('Extended-time sparse-observation-assisted reconstruction' if assisted else 'Genuine free-running: individual replicates and their means')
    return fig
def reference_credibility():
    d=csv('reference_convergence');fig,axs=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for ax,panel in zip(axs[0],('a','b','c','d')):
        for q,g in d[d.panel==panel].groupby('quantity'):
            ax.plot(g.time_s,g.value,label=q)
        ax.set_title('Reference refinement '+panel);ax.set_xlabel('Time (s)');ax.legend(fontsize=6);clean(ax)
    e=d[d.panel=='e'];axs[1,0].plot(e.time_s,e.value);axs[1,0].set_title('SGS budget identity error')
    subset=d[d.x_index.notna()]
    panels=list(subset.panel.unique());frames=[grid(subset[subset.panel==p],x='x_index',y='y_index') for p in panels]
    for ax,a,p in zip(axs[1,1:],frames,panels):maps(fig,[ax],[a],[p])
    return fig
def classical_metrics():
    d=csv('classical_apriori');keys=list(d.metric.unique());fig,axs=plt.subplots(1,len(keys),figsize=(13,4),layout='constrained')
    for ax,key in zip(axs,keys):dots(ax,d[d.metric==key],'model',('leith','clark','mixed','A0B1','full'));ax.set_title(key.replace('_',' '),fontsize=9);ax.tick_params(axis='x',rotation=25)
    return fig
def source_fields():
    d=csv('source_snapshot');models=list(d.model.unique());fig,axs=plt.subplots(2,len(models),figsize=(14,6.2),layout='constrained')
    for row,key in enumerate(('raw_value','RMS_normalized_value')):
        maps(fig,axs[row],[grid(d[d.model==m],key) for m in models],[LABELS.get(m,m) for m in models],bound=3 if row else None,normalized=bool(row))
    fig.suptitle('SGS source: raw physical (top); normalized patterns (bottom), including Clark');return fig
def field_maps():
    d=csv('rollout_snapshots');times=sorted(d.time_s.unique());models=list(d.model.unique());fig,axs=plt.subplots(len(times),len(models),figsize=(14,3.2*len(times)),layout='constrained')
    frames=[grid(g,'omega_s_inv') for _,g in d.groupby(['time_s','model'])];bound=max(np.abs(f).max() for f in frames)
    for row,t in enumerate(times):maps(fig,axs[row],[grid(d[(d.model==m)&(d.time_s==t)],'omega_s_inv') for m in models],[f'{LABELS.get(m,m)} t={t:g}' for m in models],bound=bound)
    return fig
def branch():
    d=csv('branch_allocation');fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,panel in zip(axs.ravel(),('a','b','c','d','e','f')):
        g=d[d.panel==panel].copy();g['category']=g['comparison'].fillna('')+' '+g['metric'].fillna('')+' '+g['branch'].fillna('')
        cats=list(g.category.unique())
        for i,c in enumerate(cats):
            v=g[g.category==c].value;ax.scatter(np.full(len(v),i),v,s=28)
        ax.set_xticks(range(len(cats)),cats,rotation=35,ha='right',fontsize=7);ax.set_title('Branch allocation '+panel);clean(ax)
    return fig
def spd():
    d=csv('spd_anisotropy');keys=list(d.metric.unique());fig,axs=plt.subplots(1,len(keys),figsize=(13,4),layout='constrained')
    for ax,key in zip(axs,keys):
        g=d[d.metric==key];dots(ax,g,'variant',list(g.variant.unique()),group='no_group');ax.set_title(key,fontsize=8)
    return fig
def directional():
    d=csv('directional_utility');fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,p in zip(axs,d.panel.unique()):
        g=d[d.panel==p]
        for seed,s in g.groupby('seed'):ax.plot(s.stratum.astype(str),s.relative_l2,'o-',label=str(seed))
        ax.axhline(1,color='black',ls='--',label='ISO algebraic zero');ax.set_title(p);ax.set_ylabel('Perpendicular flux relative L2');ax.legend();clean(ax)
    return fig
def independence():
    d=csv('independent_ic');d['ic']=d.ic.replace({'IC-1':'ic1','IC-2':'ic2'});fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,ic in zip(axs[0],('ic1','ic2')):
        g=d[(d.ic==ic)&d.omega.notna()];maps(fig,[ax],[grid(g,'omega','x_index','y_index')],[ic.upper()])
    for ic,g in d[d.E.notna()].groupby('ic'):axs[1,0].semilogy(g.k,g.E,label=ic)
    axs[1,0].legend();axs[1,0].set_title('Initial spectra')
    g=d[d['metric'].notna()]
    for m,s in g.groupby('variant'):axs[1,1].scatter(s.ic1,s.ic2,label=LABELS[m],color=COLORS[m])
    axs[1,1].plot([.4,1],[.4,1],'k--');axs[1,1].set(xlabel='IC-1 mean free-running error',ylabel='IC-2 mean free-running error');axs[1,1].legend();return fig
def checkpoint_history():
    d=csv('checkpoint_history');runs=list(d.run_id.unique());fig,axs=plt.subplots(len(runs),3,figsize=(13,3*len(runs)),layout='constrained',squeeze=False)
    for row,run in enumerate(runs):
        for ax,key in zip(axs[row],d.metric.unique()):
            g=d[(d.run_id==run)&(d.metric==key)];ax.plot(g.epoch,g.value);ax.axvline(g.best_epoch.iloc[0],color='r',ls='--');ax.set_title(run+' '+key);clean(ax)
    return fig
def assisted_metrics():
    d=csv('assisted_windows');keys=list(d.metric.unique());fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,ic in enumerate(('ic1','ic2')):
        for ax,key in zip(axs[row],keys):dots(ax,d[(d.ic==ic)&(d.metric==key)],'variant',('iso','A1B0','A0B1','full'),group='window');ax.set_title(ic+' '+key,fontsize=9)
    return fig
def cross_scores(zoom=False):
    d=pd.read_csv(ROOT/'data/tables/cross_regime_classical_neural_data.csv');fig,axs=plt.subplots(2,3,figsize=(10.5,6.8),sharey='row',layout='constrained')
    for col,group in enumerate((('baseline','filter_small','filter_large'),('baseline','half_Re','double_Re'),('baseline','forced_strong'))):
        for row,key in enumerate(('Pi_Pearson_apriori','omega_mean_free_running')):
            ax=axs[row,col]
            for j,r in enumerate(group):
                for x,m in enumerate(('leith','clark','mixed','A0B1')):
                    v=d[(d.regime==r)&(d.model==m)][key].to_numpy();pos=x+(j-(len(group)-1)/2)*.18;color=RC[REGIMES.index(r)];marker=('o','s','^')[j]
                    ax.scatter(pos+np.linspace(-.035,.035,len(v)),v,facecolors='none',edgecolors=color,marker='D' if len(v)==1 else marker,s=27)
                    ax.errorbar(pos,v.mean(),yerr=v.std(ddof=1) if len(v)>1 else None,fmt='D' if len(v)==1 else marker,color=color,capsize=2.4,ms=5.7)
                ax.scatter([],[],color=color,marker=marker,label=r)
            ax.set_xticks(range(4),('Leith','Clark','Mixed','A0B1'));ax.set_ylim((0,1.04) if row==0 else (0,1.08 if zoom else max(4.05,1.04*max(g.mean()+g.std(ddof=1) if len(g)>1 else g.iloc[0] for _,g in d.groupby(['regime','model'])[key]))));clean(ax)
            if row==0:ax.set_title(('Filter widths','Fixed-scale Re','Periodic forcing')[col]);ax.legend(fontsize=7,frameon=False)
            if col==0:ax.set_ylabel('A-priori Pi Pearson' if row==0 else 'Mean free-running vorticity relative L2')
    return fig
def cross_curves():
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,r in zip(axs.ravel(),REGIMES):
        for seed in (2026,2027,2028):
            if r=='baseline':p=ROOT/f'results/frozen/neural/ic1/A0B1_{seed}_free_running/timewise.csv'
            else:
                base=ROOT/'results/frozen/cross_regime';p=next(p for p in base.rglob('per_time_metrics.csv') if r in p.parts and 'ic1' in p.parts and 'A0B1' in p.parts and 'free_running_continuation' in p.parts and f'seed_{seed}' in p.parts)
            d=pd.read_csv(p);ax.plot(d.time_s,d.omega_relative_l2,label=str(seed))
        ax.set_title(r);ax.set_xlabel('Time (s)');ax.set_ylabel('Vorticity relative L2');ax.legend(fontsize=8);clean(ax)
    return fig
def cross_reference():
    records=read_json(ROOT/'manifests/datasets.json');refs={}
    for r in REGIMES:
        with np.load(ROOT/records[r+'/ic1']['physical']['path'],allow_pickle=False) as z:refs[r]={k:z[k].copy() for k in ('omega','time_s','K','Z')}
    fig,axs=plt.subplots(3,6,figsize=(15,8),layout='constrained');base=refs['baseline']['omega'][-1];diffs=[refs[r]['omega'][-1]-base for r in REGIMES[1:]];bound=max(np.abs(a).max() for a in diffs)
    maps(fig,[axs[0,0]],[base],['Baseline']);maps(fig,axs[0,1:],diffs,list(REGIMES[1:]),bound=bound)
    for col,r in enumerate(REGIMES):
        for row,key in ((1,'K'),(2,'Z')):axs[row,col].plot(refs[r]['time_s'],refs[r][key]);axs[row,col].set_title(r+' '+key);axs[row,col].set_xlabel('Time (s)');clean(axs[row,col])
    return fig

BUILDERS={'Fig01_method_information_access':method,'Fig02_reference_credibility':reference_credibility,'Fig03_factorial_attribution':factorial,'Fig04_signed_transfer_structure':pi_fields,'Fig05_classical_neural_tradeoff':tradeoff,'Fig06_free_running_full_scale':rollout,'Fig07_observation_assisted_10s':lambda:rollout(assisted=True),'Fig08_classical_neural_cross_regime':cross_scores,
          'FigS01_classical_a_priori_metrics':classical_metrics,'FigS02_source_fields_with_clark':source_fields,'FigS03_free_running_field_maps':field_maps,'FigS04_individual_neural_free_running_seeds':lambda:rollout(classical=False),'FigS05_branch_allocation':branch,'FigS06_spd_anisotropy':spd,'FigS07_directional_anisotropy_utility':directional,'FigS08_ic_1_ic_2_independence':independence,'FigS09_checkpoint_audit':checkpoint_history,'FigS10_assisted_late_window_fidelity':assisted_metrics,'FigS12_normalized_pi_density':density_only,'FigS11_reference_regime_differences':cross_reference,'FigS12_cross_regime_free_running':cross_curves,'FigS13_cross_regime_free_running_zoom':lambda:cross_scores(zoom=True)}
def reproduce(output,selection='all',dpi=600):
    out=new_output(output);out.mkdir(parents=True);(out/'published_exact').mkdir();(out/'redrawn').mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    selected=read_json(ROOT/'manifests/figures.json')['selected_files'];record=[]
    for name in selected:
        if selection=='main' and name.startswith('FigS'):continue
        if selection=='supplement' and not name.startswith('FigS'):continue
        src=ROOT/'figures/published'/name;shutil.copyfile(src,out/'published_exact'/name)
        builder=BUILDERS.get(Path(name).stem)
        if builder is None:
            from .publication_evidence import generate
            extra=out/'new_evidence'
            if not extra.exists():generate(extra,dpi)
            for ext in ('pdf','svg','png'):shutil.copy2(extra/(Path(name).stem+'.'+ext),out/'redrawn'/(Path(name).stem+'.'+ext))
            record.append(dict(figure=name,archival_sha256=sha256(src),numeric_redraw=True,pixel_identical_claim=False))
            continue
        fig=builder()
        for ext in ('pdf','svg','png'):fig.savefig(out/'redrawn'/f'{Path(name).stem}.{ext}',bbox_inches='tight',dpi=dpi)
        plt.close(fig);record.append(dict(figure=name,archival_sha256=sha256(src),redrawn_png_sha256=sha256(out/'redrawn'/f'{Path(name).stem}.png'),numeric_redraw=True,pixel_identical_claim=False))
        print('REDRAW',name)
    report=dict(figures=record,plotting_only=True,model_inference=False,source_manifest_sha256=sha256(ROOT/'manifests/figures.json'))
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return out
