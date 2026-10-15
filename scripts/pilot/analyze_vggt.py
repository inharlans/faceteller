"""Cohort calibration, distribution checks and repeated-capture reliability."""
import csv,json,sys,collections
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.pilot.scoring import boundary_scores,estimate_boundaries,repeat_sigma
from facework.pilot.depth import FEATURES
D=ROOT/'data/derived/pilot-20261014';OUT=ROOT/'outputs/pilot-20261014';OUT.mkdir(exist_ok=True,parents=True)

def rows(phase,size=518):
    p=D/f'{phase}_{size}.jsonl'
    if not p.exists():return []
    attempts=[json.loads(l) for l in p.read_text().splitlines()]
    # Retries are additional attempts, never additional participants/captures.
    return list({r['job']:r for r in attempts}.values())

def icc(a):
    a=np.asarray(a);n,k=a.shape
    if n<2 or k<2:return None
    msb=k*((a.mean(1)-a.mean())**2).sum()/(n-1)
    msw=((a-a.mean(1,keepdims=True))**2).sum()/(n*(k-1))
    return float((msb-msw)/(msb+(k-1)*msw))

def run(size=518):
    base=rows('cohort',size);repeat=rows('repeat',size);controls=rows('controls',size)
    good={r['subject_id']:r for r in base if r['status']=='measured'}
    groups={s:{'L1':r} for s,r in good.items()}
    for r in repeat:
        if r['status']=='measured' and r['subject_id'] in groups:groups[r['subject_id']][r['light']]=r
    repeated={s:g for s,g in groups.items() if len(g)==3}
    report={'cohort_attempted':len(base),'cohort_measured':len(good),'cohort_failures':[{'subject_id':r['subject_id'],'reason':r.get('error',r.get('reason'))} for r in base if r['status']!='measured'],
            'repeat_attempted':len(repeat),'complete_three_light_participants':len(repeated),'input_size':size,
            'boundaries_scope':'all available cohort participants, exploratory population reference; not independent test accuracy',
            'metrics_are_reconstructed_proxies_not_ground_truth':True,'features':{},'controls':{}}
    ts=[r['inference_seconds'] for r in base if 'inference_seconds' in r]
    if ts:report['vggt_warm_timing']={'n':len(ts)-1,'mean':float(np.mean(ts[1:])),'median':float(np.median(ts[1:])),'p95':float(np.quantile(ts[1:],.95)),'max':max(ts[1:]),'first_request':ts[0],'peak_gib':max(r.get('peak_allocated_gib',0) for r in base),'scope':'tensor preparation + GPU network + depth-to-world unprojection; not CNN/HTTP/DB'}
    train={s:r for s,r in good.items() if r['split']=='train'}
    for f in FEATURES:
        a=np.array([r['values'][f] for r in good.values()]);cfg=estimate_boundaries(a)
        sw=stats.shapiro(a);normcuts=stats.norm.ppf([.25,.75],loc=a.mean(),scale=a.std(ddof=1))
        labels=np.digitize(a,[cfg['lower'],cfg['upper']])
        gaussian_labels=np.digitize(a,normcuts)
        cfg.update(normality={'shapiro_w':float(sw.statistic),'shapiro_p':float(sw.pvalue),'skew':float(stats.skew(a)),'excess_kurtosis':float(stats.kurtosis(a))},
                   gaussian_assumed_boundaries=normcuts.tolist(),empirical_vs_gaussian_label_agreement=float(np.mean(labels==gaussian_labels)),
                   cohort_class_counts={k:int((labels==i).sum()) for i,k in enumerate(['low','medium','high'])})
        rng=np.random.default_rng(20261014)
        bootstrap=np.quantile(rng.choice(a,(2000,len(a)),replace=True),[.25,.75],axis=1)
        cfg['boundary_bootstrap95']=np.quantile(bootstrap,[.025,.975],axis=1).T.tolist()
        pop=[boundary_scores(v,cfg['lower'],cfg['upper'],cfg['sigma_population']) for v in a]
        cfg['population_sigma_score']={'mean_assigned_score':float(np.mean([r['score'] for r in pop])),
              'hard_label_argmax_disagreement':sum(r['label']!=max(r['scores'],key=r['scores'].get) for r in pop),
              'middle_score_at_interval_midpoint':boundary_scores((cfg['lower']+cfg['upper'])/2,cfg['lower'],cfg['upper'],cfg['sigma_population'])['scores']['medium']}
        if repeated:
            mat=np.array([[g[l]['values'][f] for l in ['L1','L2','L3']] for g in repeated.values()]);sigma=repeat_sigma(mat)
            cfg['sigma_repeat_all']=sigma
            calib_mat=np.array([[g[l]['values'][f] for l in ['L1','L2','L3']] for s,g in repeated.items() if good[s]['split']=='train'])
            cfg['repeat_sigma_train_participants']=len(calib_mat)
            cfg['sigma_repeat_train']=repeat_sigma(calib_mat) if len(calib_mat)>1 else None
            lab=np.digitize(mat,[cfg['lower'],cfg['upper']])
            cfg['repeat']={'n':len(mat),'icc_1_1':icc(mat),'mean_abs_change_L2_L3_from_L1':float(np.abs(mat[:,1:]-mat[:,:1]).mean()),'label_agreement_L1_L2':float(np.mean(lab[:,0]==lab[:,1])),'label_agreement_L1_L3':float(np.mean(lab[:,0]==lab[:,2])),'all_three_label_agreement':float(np.mean((lab==lab[:,:1]).all(1))), 'mean_shift_L2':float((mat[:,1]-mat[:,0]).mean()),'mean_shift_L3':float((mat[:,2]-mat[:,0]).mean())}
            cfg['repeat_selective_agreement']={}
            evaluation_subjects=[s for s in repeated if good[s]['split']!='train']
            for scope,subids in [('all_40_descriptive',list(repeated)),('heldout_nontrain',evaluation_subjects)]:
                cfg['repeat_selective_agreement'][scope]={}
                cuts_eval=np.quantile([r['values'][f] for r in train.values()],[.25,.75]) if scope=='heldout_nontrain' else [cfg['lower'],cfg['upper']]
                for threshold in [.6,.7,.8]:
                    accepted=0;total=0;agree=0
                    for sid in subids:
                        vals=[repeated[sid][l]['values'][f] for l in ['L1','L2','L3']]
                        ss=[boundary_scores(v,float(cuts_eval[0]),float(cuts_eval[1]),cfg['sigma_repeat_train'] or sigma) for v in vals]
                        for j in [1,2]:
                            total+=1
                            if min(ss[0]['score'],ss[j]['score'])>=threshold:
                                accepted+=1;agree+=ss[0]['label']==ss[j]['label']
                    cfg['repeat_selective_agreement'][scope][str(threshold)]={'boundary_source':'train_only_276' if scope=='heldout_nontrain' else 'all_cohort_400','people':len(subids),'pairs':total,'accepted_pairs':accepted,'pair_coverage':accepted/total if total else None,'label_agreement_on_accepted_pairs':agree/accepted if accepted else None}
            for sigma_name,sigma_val in [('population',cfg['sigma_population']),('repeat',sigma)]:
                scored=[boundary_scores(v,cfg['lower'],cfg['upper'],sigma_val) for v in a]
                cfg[sigma_name+'_sigma_coverage_at_060']=float(np.mean([v['score']>=.6 for v in scored]))
        if train:
            tc=estimate_boundaries([r['values'][f] for r in train.values()])
            cfg['train_only_calibration']=tc
            test=[r['values'][f] for r in good.values() if r['split']=='test']
            cfg['test_class_counts_using_train_boundaries']=dict(zip(['low','medium','high'],np.bincount(np.digitize(test,[tc['lower'],tc['upper']]),minlength=3).tolist()))
        report['features'][f]=cfg
    for r in controls:
        if r['status']!='measured' or r['subject_id'] not in good:continue
        name=f'{r["mode"]}:{",".join(r["cameras"])}:margin{r["margin"]}'
        rec=report['controls'].setdefault(name,{f:[] for f in FEATURES})
        for f in FEATURES:rec[f].append([good[r['subject_id']]['values'][f],r['values'][f]])
    for name,fs in report['controls'].items():
        for f,pairs in list(fs.items()):
            a=np.asarray(pairs);cfg=report['features'][f];cuts=[cfg['lower'],cfg['upper']]
            fs[f]={'n':len(pairs),'mean_abs_delta':float(np.abs(a[:,0]-a[:,1]).mean()),'label_agreement':float(np.mean(np.digitize(a[:,0],cuts)==np.digitize(a[:,1],cuts)))}
    # Cross-method agreement is not accuracy: earlier TK shape has no 3D ground truth either.
    old=json.loads((ROOT/'data/derived/kface/depth.json').read_text())['subjects']
    for f in FEATURES:
        pairs=[(r['values'][f],old[s]['sessions']['L1'][f]) for s,r in good.items() if s in old and 'L1' in old[s]['sessions']]
        if len(pairs)>3:
            a=np.array(pairs);rho=stats.spearmanr(a[:,0],a[:,1])
            report['features'][f]['comparison_TK12views']={'n':len(pairs),'spearman':float(rho.statistic),'pearson':float(np.corrcoef(a.T)[0,1]),'is_ground_truth':False}
    (OUT/'depth_distribution_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (D/'boundary_config.json').write_text(json.dumps({'input_size':size,'features':report['features'],'version':'pilot-20261014','status':'exploratory_not_validated_3D_truth'},indent=2))
    fields=['subject_id','split']+[f+'_'+x for f in FEATURES for x in ['ratio','label','score_population','score_repeat']]
    with (D/'cohort_scores.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        wr=csv.DictWriter(fp,fields);wr.writeheader()
        for s,r in good.items():
            row={'subject_id':s,'split':r['split']}
            for f in FEATURES:
                c=report['features'][f];v=r['values'][f];p=boundary_scores(v,c['lower'],c['upper'],c['sigma_population'])
                row.update({f+'_ratio':v,f+'_label':p['label'],f+'_score_population':p['score'],f+'_score_repeat':boundary_scores(v,c['lower'],c['upper'],c['sigma_repeat_all'])['score'] if c.get('sigma_repeat_all') else ''})
            wr.writerow(row)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,f in zip(axes.flat,FEATURES):
        a=np.array([r['values'][f] for r in good.values()]);c=report['features'][f]
        ax.hist(a,bins=25,density=True,alpha=.55,color='#326b9d');x=np.linspace(a.min(),a.max(),400)
        ax.plot(x,stats.norm.pdf(x,c['mean'],c['sigma_population']),color='#d46d34',label='Gaussian fit')
        for b in [c['lower'],c['upper']]:ax.axvline(b,color='#222',ls='--')
        ax.set(title=f.replace('_',' '),xlabel='Relative projection / outer-eye distance',ylabel='Density');ax.legend()
    fig.suptitle(f'VGGT real-photo cohort: {len(good)} people; dashed lines = empirical P25/P75')
    fig.savefig(OUT/'depth_distributions.png',dpi=160);fig.savefig(OUT/'depth_distributions.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='controls'},ensure_ascii=False,indent=2))
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 518)
