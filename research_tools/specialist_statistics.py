"""Paired grouped specialist analysis and descriptive frozen TEST graphics; no inference."""
from pathlib import Path
import csv,json,hashlib,itertools,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';S=D/'normal_benign_specialist_v1';O=S/'statistics';O.mkdir(exist_ok=False);F=O/'figures';F.mkdir()
C=['Benign','Malignant','Normal','PCOS'];names=['unchanged_main','plain_specialist','mined_specialist'];titles=['Unchanged main','Plain specialist','Hard-mined specialist'];colors=['#285A8E','#D17A22','#8A4267'];rng=np.random.default_rng(2627)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
rows={r['record_id']:r for r in csv.DictReader((B/'Classification/metadata/dataset_manifest.csv').open())};z=np.load(S/'selection_predictions.npz',allow_pickle=False);ids=z['record_ids'];y=z['true_class_indices'];records=[rows[str(id)] for id in ids];np.testing.assert_array_equal(y,[C.index(r['class_name']) for r in records]);np.testing.assert_array_equal(z['classes'],C);assert len(y)==2112

def grouping(records):
 parent=list(range(len(records)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 seen={}
 for i,r in enumerate(records):
  for key in ['image_group','pixel_sha256','sha256']:
   item=(key,r[key])
   if item in seen:parent[find(i)]=find(seen[item])
   else:seen[item]=i
 return np.unique([find(i) for i in range(len(records))],return_inverse=True)[1]

def grouped_cm(truth,pred,g):
 cm=np.zeros((int(g.max())+1,4,4),int);np.add.at(cm,(g,truth,pred),1);return cm

def metrics(cm):
 diag=np.diagonal(cm,axis1=-2,axis2=-1);row=cm.sum(-1);col=cm.sum(-2);total=cm.sum((-1,-2));rec=np.divide(diag,row,out=np.full(diag.shape,np.nan),where=row>0);prec=np.divide(diag,col,out=np.zeros(diag.shape),where=col>0);tn=total[...,None]-row-col+diag;neg=total[...,None]-row;spec=np.divide(tn,neg,out=np.full(diag.shape,np.nan),where=neg>0);f1=np.divide(2*diag,row+col,out=np.zeros(diag.shape),where=(row+col)>0)
 return {'balanced_accuracy':rec.mean(-1),'macro_f1':f1.mean(-1),'accuracy':diag.sum(-1)/total,'recall':rec,'precision':prec,'specificity':spec}

def ci(a):return np.nanpercentile(a,[2.5,97.5],axis=0).tolist()
def savefig(fig,name):
 fig.tight_layout();fig.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def draw_cm(ax,m,title):
 normalized=m/np.maximum(m.sum(1,keepdims=True),1);ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
 for i,j in itertools.product(range(4),repeat=2):ax.text(j,i,f'{m[i,j]}\n{100*normalized[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if normalized[i,j]>.55 else 'black')
 ax.set(xticks=range(4),yticks=range(4),xticklabels=C,yticklabels=C,title=title,xlabel='Predicted class',ylabel='Provided class');ax.tick_params(axis='x',rotation=35)

# Tests of core grouped statistics: identical predictions must yield zero paired effect.
tg=np.array([0,0,1,1]);ty=np.array([0,1,2,3]);tc=grouped_cm(ty,ty,tg);assert metrics(tc.sum(0))['balanced_accuracy']==1.;assert np.array_equal(tc-tc,np.zeros_like(tc))
g=grouping(records);G=int(g.max())+1;draws=rng.multinomial(G,np.ones(G)/G,size=2500);cms={name:grouped_cm(y,z[name],g) for name in names};point={name:metrics(cm.sum(0)) for name,cm in cms.items()};boot={}
with threadpool_limits(limits=2):
 for name,cm in cms.items():boot[name]=metrics(np.tensordot(draws,cm,axes=(1,0)))
report={'scope':'Exploratory SELECT specialist comparison; repeated SELECT experimentation is not corrected by these tests. Representation learned from allFIT; folds were head-mining only. Bootstrap/permutation clusters join declared similarity and matching byte/pixel groups, not patients. No new TEST or calibration inference.','selection_records':len(y),'selection_groups':G,'bootstrap_repetitions':2500,'permutation_repetitions':5000,'bootstrap_method':'Paired percentile cluster bootstrap: sample whole connected image groups with replacement, preserving their record multiplicities.','seed':2627,'class_order':C,'models':{},'contrasts':{}}
for name in names:
 report['models'][name]={k:np.asarray(v).tolist() for k,v in point[name].items()};report['models'][name]['group_bootstrap_95_ci']={k:ci(v) for k,v in boot[name].items()}
contrasts=[('unchanged_main','plain_specialist'),('unchanged_main','mined_specialist'),('plain_specialist','mined_specialist')]
for a,b in contrasts:
 delta=float(point[b]['balanced_accuracy']-point[a]['balanced_accuracy']);support=np.bincount(y,minlength=4);contribution=((np.diagonal(cms[b],axis1=1,axis2=2)-np.diagonal(cms[a],axis1=1,axis2=2))/support).sum(1)/4;np.testing.assert_allclose(contribution.sum(),delta,atol=1e-14)
 null=(rng.integers(0,2,size=(5000,G))*2-1)@contribution;p=float((1+np.count_nonzero(abs(null)>=abs(delta)-1e-14))/5001)
 report['contrasts'][a+'__to__'+b]={'balanced_accuracy_difference':delta,'paired_group_bootstrap_95_ci':ci(boot[b]['balanced_accuracy']-boot[a]['balanced_accuracy']),'macro_f1_difference':float(point[b]['macro_f1']-point[a]['macro_f1']),'macro_f1_paired_group_bootstrap_95_ci':ci(boot[b]['macro_f1']-boot[a]['macro_f1']),'group_swap_two_sided_p':p,'interpretation':'Exploratory cluster-swap symmetry test; not independent confirmatory evidence.'}
keys=list(report['contrasts']);order=sorted(keys,key=lambda k:report['contrasts'][k]['group_swap_two_sided_p']);running=0.
for i,key in enumerate(order):running=max(running,min(1.,(len(keys)-i)*report['contrasts'][key]['group_swap_two_sided_p']));report['contrasts'][key]['holm_adjusted_p_three_contrasts']=running
with (O/'selection_metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','metric','estimate','CI_low','CI_high'])
 for name in names:
  for metric in ['balanced_accuracy','macro_f1','accuracy']:w.writerow([name,metric,float(point[name][metric]),*ci(boot[name][metric])])
with (O/'selection_per_class.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','class','support','recall','CI_low','CI_high','precision','specificity'])
 for name in names:
  lo,hi=ci(boot[name]['recall'])
  for i,c in enumerate(C):w.writerow([name,c,int((y==i).sum()),point[name]['recall'][i],lo[i],hi[i],point[name]['precision'][i],point[name]['specificity'][i]])
with (O/'selection_per_source_class.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','source','class','support','correct','recall'])
 for name in names:
  for source in sorted({r['source'] for r in records}):
   for i,c in enumerate(C):
    idx=np.array([r['source']==source for r in records])&(y==i);n=int(idx.sum());correct=int((z[name][idx]==i).sum());w.writerow([name,source,c,n,correct,correct/n if n else 'NA'])
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for ax,name,title in zip(axes,names,titles):draw_cm(ax,cms[name].sum(0),title)
fig.suptitle('SELECT: matched routing comparison (counts and within-class percentages)',y=1.03);savefig(fig,'01_selection_confusion_matrices')
fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4)
for k,name in enumerate(names):
 lo,hi=np.asarray(ci(boot[name]['recall']));v=point[name]['recall'];ax.errorbar(x+(k-1)*.16,v,yerr=[v-lo,hi-v],fmt='o',capsize=3,color=colors[k],label=titles[k])
ax.set(xticks=x,xticklabels=C,ylabel='Recall',ylim=(.65,1.01),title='SELECT per-class recall; paired image-group bootstrap 95% intervals');ax.legend();savefig(fig,'02_selection_class_recall')
fig,axes=plt.subplots(1,2,figsize=(10,4.6))
for ax,metric,title in zip(axes,['balanced_accuracy','macro_f1'],['Balanced accuracy','Macro F1']):
 for k,name in enumerate(names):
  lo,hi=ci(boot[name][metric]);v=float(point[name][metric]);ax.errorbar(k,v,yerr=[[max(v-lo,0)],[max(hi-v,0)]],fmt='o',capsize=5,color=colors[k]);ax.text(k,v+.004,f'{100*v:.2f}%',ha='center')
 ax.set(xticks=range(3),xticklabels=['Main','Plain','Mined'],ylabel=title,title=title+' (SELECT)')
fig.suptitle('Group-bootstrap 95% intervals; no TEST model selection');savefig(fig,'03_selection_metrics')
fig,ax=plt.subplots(figsize=(9,4.4))
for i,(key,v) in enumerate(report['contrasts'].items()):
 delta=v['balanced_accuracy_difference']*100;lo,hi=np.array(v['paired_group_bootstrap_95_ci'])*100;ax.errorbar(delta,i,xerr=[[max(delta-lo,0)],[max(hi-delta,0)]],fmt='o',capsize=4);ax.text(hi+.04,i,f"Holm p={v['holm_adjusted_p_three_contrasts']:.3f}",va='center',fontsize=9)
ax.axvline(0,color='grey',ls='--');ax.set(yticks=range(3),yticklabels=['Plain − main','Mined − main','Mined − plain'],xlabel='Balanced-accuracy difference (percentage points)',title='Paired group-bootstrap effects; exploratory group-swap tests');savefig(fig,'04_paired_effects')
fig,axes=plt.subplots(1,2,figsize=(10,4.5));pair=np.isin(y,[0,2]);binary=(y[pair]==2).astype(int);pbase=z['main_probabilities'][pair,2]/np.maximum(z['main_probabilities'][pair,0]+z['main_probabilities'][pair,2],1e-9)
for name,title,color,p in zip(names,titles,colors,[pbase,z['plain_specialist_normal_probability'][pair],z['mined_specialist_normal_probability'][pair]]):
 fpr,tpr,_=roc_curve(binary,p);precision,recall,_=precision_recall_curve(binary,p);axes[0].plot(fpr,tpr,color=color,label=f'{title}: AUC {roc_auc_score(binary,p):.3f}');axes[1].plot(recall,precision,color=color,label=f'{title}: AP {average_precision_score(binary,p):.3f}')
axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Benign/Normal subset ROC');axes[1].set(xlabel='Normal recall',ylabel='Normal precision',title='Benign/Normal subset precision–recall')
for ax in axes:ax.legend(fontsize=8)
fig.suptitle('Uncalibrated specialist scores, SELECT two-class subset; not four-class pipeline AUC');savefig(fig,'05_specialist_binary_curves')
fig,ax=plt.subplots(figsize=(8,4.5));xx=np.arange(4)
for k,name in enumerate(names[1:]):
 changes=z[name]!=z['unchanged_main'];values=[int((changes&(y==i)).sum()) for i in range(4)];ax.bar(xx+(k-.5)*.32,values,width=.32,color=colors[k+1],label=titles[k+1])
ax.set(xticks=xx,xticklabels=C,ylabel='Records switched Benign → Normal',title='Routing impacts by provided class; only Normal gains can become correct');ax.legend();savefig(fig,'06_routing_impacts')
# Historical TEST: reanalyse frozen scores only, not an optimization endpoint.
path=B/'Preserved_Best_Completed_Version_2026_11_22/results/ensemble_control/evaluation/test_predictions.csv';test=list(csv.DictReader(path.open()));tr=[rows[r['record_id']] for r in test];truth=np.array([C.index(r['true_class']) for r in test]);pred=np.array([C.index(r['predicted_class']) for r in test]);p=np.array([[float(r[c]) for c in C] for r in test]);np.testing.assert_array_equal(p.argmax(1),pred);tg=grouping(tr);TC=grouped_cm(truth,pred,tg);T=TC.shape[0];tdraw=rng.multinomial(T,np.ones(T)/T,size=2500)
with threadpool_limits(limits=2):tb=metrics(np.tensordot(tdraw,TC,axes=(1,0)))
tpoint=metrics(TC.sum(0));confidence=p.max(1);correct=(pred==truth).astype(float);bins=np.minimum((confidence*10).astype(int),9);cnt=np.zeros((T,10));cs=np.zeros_like(cnt);ps=np.zeros_like(cnt);np.add.at(cnt,(tg,bins),1);np.add.at(cs,(tg,bins),correct);np.add.at(ps,(tg,bins),confidence)
with threadpool_limits(limits=2):bc=tdraw@cnt;bs=tdraw@cs;bp=tdraw@ps
bece=abs(bs-bp).sum(1)/bc.sum(1);nll=-np.log(np.maximum(p[np.arange(len(truth)),truth],1e-7));brier=((p-np.eye(4)[truth])**2).sum(1);gnll=np.bincount(tg,weights=nll,minlength=T);gbrier=np.bincount(tg,weights=brier,minlength=T)
with threadpool_limits(limits=2):bnll=(tdraw@gnll)/bc.sum(1);bbrier=(tdraw@gbrier)/bc.sum(1)
test_report={'scope':'Descriptive reanalysis of frozen historical TEST probabilities; no new image inference or model choice. New bootstrap intervals do not replace archived original intervals.','records':len(truth),'groups':T,'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':{k:np.asarray(v).tolist() for k,v in tpoint.items()},'group_bootstrap_95_ci':{k:ci(v) for k,v in tb.items()},'macro_roc_auc_ovr':float(roc_auc_score(truth,p,multi_class='ovr',average='macro')),'macro_average_precision_ovr':float(np.mean([average_precision_score(truth==i,p[:,i]) for i in range(4)])),'nll':float(nll.mean()),'nll_95_ci':ci(bnll),'brier':float(brier.mean()),'brier_95_ci':ci(bbrier),'ece10':float(abs(cs.sum(0)-ps.sum(0)).sum()/len(truth)),'ece10_95_ci':ci(bece)}
np.testing.assert_allclose(tpoint['balanced_accuracy'],.8954576875376714,atol=1e-14)
# Covariance-based descriptive decomposition, rather than assuming which class dominates variance.
components=tb['recall']/4;ba=tb['balanced_accuracy'];covariance=np.array([np.cov(components[:,i],ba,ddof=1)[0,1] for i in range(4)]);variance=np.var(ba,ddof=1);np.testing.assert_allclose(covariance.sum(),variance,rtol=1e-8);test_report['bootstrap_variance_covariance_contributions']=dict(zip(C,(covariance/variance).tolist()));test_report['variance_note']='Covariance allocation under this bootstrap; may include cross-class covariance, not causal or unique class attribution.'
fig,ax=plt.subplots(figsize=(6.3,5.3));draw_cm(ax,TC.sum(0),'Frozen historical TEST; unchanged ensemble');savefig(fig,'07_historical_test_confusion')
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for i,c in enumerate(C):
 fpr,tpr,_=roc_curve(truth==i,p[:,i]);prec,rec,_=precision_recall_curve(truth==i,p[:,i]);axes[0].plot(fpr,tpr,label=f'{c} AUC {roc_auc_score(truth==i,p[:,i]):.3f}');axes[1].plot(rec,prec,label=f'{c} AP {average_precision_score(truth==i,p[:,i]):.3f}')
axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set(xlabel='False-positive rate',ylabel='Recall',title='Frozen TEST one-vs-rest ROC');axes[1].set(xlabel='Recall',ylabel='Precision',title='Frozen TEST one-vs-rest precision–recall')
for ax in axes:ax.legend(fontsize=8)
fig.suptitle('Descriptive historical curves; no new threshold selection');savefig(fig,'08_historical_test_curves')
fig,axes=plt.subplots(1,2,figsize=(10,4.6));counts=cnt.sum(0);mask=counts>0;avconf=ps.sum(0)[mask]/counts[mask];avacc=cs.sum(0)[mask]/counts[mask];axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].plot(avconf,avacc,'o-',label=f'ECE10={test_report["ece10"]:.3f}');axes[0].set(xlabel='Mean maximum class probability',ylabel='Observed correctness',title='Frozen TEST reliability (10 fixed bins)');axes[0].legend();axes[1].hist(confidence,bins=np.linspace(0,1,11),edgecolor='white');axes[1].set(xlabel='Maximum probability',ylabel='Record count',title='Frozen TEST confidence distribution');savefig(fig,'09_historical_reliability')
sources=sorted({r['source'] for r in tr});heat=np.full((len(sources),4),np.nan);support=np.zeros_like(heat,dtype=int)
for j,s in enumerate(sources):
 for i,c in enumerate(C):
  idx=np.array([r['source']==s for r in tr])&(truth==i);support[j,i]=idx.sum()
  if idx.any():heat[j,i]=(pred[idx]==i).mean()
fig,ax=plt.subplots(figsize=(9,5));cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee');ax.imshow(heat,vmin=0,vmax=1,cmap=cmap,aspect='auto')
for j,i in itertools.product(range(len(sources)),range(4)):ax.text(i,j,f'{100*heat[j,i]:.1f}%\nn={support[j,i]}' if support[j,i] else 'Not available\nn=0',ha='center',va='center',fontsize=9,color='white' if heat[j,i]>.6 else 'black')
ax.set(xticks=range(4),xticklabels=C,yticks=range(len(sources)),yticklabels=sources,title='Frozen TEST class recall by collection; absent classes are unavailable');savefig(fig,'10_historical_source_class_recall')
folds=json.loads((S/'results.json').read_text())['folds'];fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(range(1,6),[f['held'] for f in folds],color='#285A8E');ax.set(xticks=range(1,6),xlabel='Head-level fold',ylabel='Held FIT records',title='Five-fold mining assignments; representation already saw all FIT');savefig(fig,'11_head_mining_fold_sizes')
for name in names:np.savetxt(O/(name+'_confusion_matrix.csv'),cms[name].sum(0),delimiter=',',fmt='%d',header=','.join(C))
(O/'specialist_statistics.json').write_text(json.dumps(report,indent=2,allow_nan=False));(O/'historical_test_descriptive.json').write_text(json.dumps(test_report,indent=2,allow_nan=False));np.savez_compressed(O/'bootstrap_metric_draws.npz',**{name+'_'+metric:value for name,data in boot.items() for metric,value in data.items()},historical_test_recall=tb['recall'],historical_test_balanced_accuracy=tb['balanced_accuracy'])
text='SPECIALIST STATISTICAL REPORT\n\nPrimary comparison uses all2112SELECTrecords, with connectedimagegroup resampling. This is exploratory after repeated selection reuse, not independent verification of a clinical claim. No calibrated routed-model probabilities exist.\n\n'
for name in names:
 lo,hi=report['models'][name]['group_bootstrap_95_ci']['balanced_accuracy'];text+=f'{name}: balanced accuracy{100*float(point[name]["balanced_accuracy"]):.4f}%, group-bootstrap95%CI[{100*lo:.4f},{100*hi:.4f}]%.\n'
for key,v in report['contrasts'].items():
 lo,hi=v['paired_group_bootstrap_95_ci'];text+=f'{key}: delta{100*v["balanced_accuracy_difference"]:.4f}pp, paired95%CI[{100*lo:.4f},{100*hi:.4f}]pp, group-swapP={v["group_swap_two_sided_p"]:.5f}, Holm3P={v["holm_adjusted_p_three_contrasts"]:.5f}.\n'
text+='\nDecision: neither specialist improves balanced accuracy; no promotion or threshold tuning. Only Benign→Normal changes are allowed, so Malignant/PCOS recall is algebraically unchanged. Wholepipelinekfold was NOT performed. Fivefolds are FIT-only conditional head mining; frozen backbone saw allFIT, and inherited base weights were calculated onFIT.\n\nFrozen historical TEST reanalysis is descriptive. Its probabilities/images did not train this specialist or select gates. Global AUC/ECE are not evidence that every class/source calibrates well. Source n=0 cells are NA; tiny supports require caution. NewbootstrapCIs use2500draws and do not overwrite original archivedCIs. No ordinary independent-record McNemar/DeLong significance claim is made because repeated/groupedimages violate that simple independence assumption. Holm3 covers only the current three contrasts, not earlier selection search. Bootstrap/permutation intervals are not clinical validation.\n\nMethods: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html . Custom bootstrap draws whole connected imagegroups jointly across models; custom cluster-swap test exchanges paired predictions bygroup.\n\nArtifacts:11figures each inPNG/PDF; CSVmetrics,perclass,per-source,confusionmatrices; JSONstatistics; bootstrapdraws. No source ultrasound images included in figure exports.\n'
(O/'STATISTICAL_REPORT.txt').write_text(text);print(text,flush=True);print('HISTORICAL_BOOTSTRAP_VARIANCE_CONTRIBUTIONS',test_report['bootstrap_variance_covariance_contributions'],flush=True)
