"""Descriptive paired analysis of eight already-frozen TEST prediction files."""
from pathlib import Path
import ast,csv,io,json,zipfile,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');D=B/'Accuracy_Development_2026_11_22';O=D/'normal_benign_specialist_v1/statistics/historical_project';O.mkdir(exist_ok=False);F=O/'figures';F.mkdir()
# Reuse exactly the tested mathematical helpers, without running the specialist plot script.
source=D/'tools/specialist_statistics.py';tree=ast.parse(source.read_text());definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['grouping','grouped_cm','metrics','ci']];assert len(definitions)==4;exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'))
C=['Benign','Malignant','Normal','PCOS'];names=['original_baseline','fusion','fusion_refined','image_only','localized_no_morphology','fusion_zero_morphology','ensemble_anatomy','ensemble_control'];labels=['Original hybrid','Fusion','Refined fusion','Image only','Global + ROI','Zero-morphology control','Anatomy ensemble','Control ensemble'];path=B/'Preserved_Best_Completed_Version_2026_11_22/Final_Project_Results.zip';assert hashlib.sha256(path.read_bytes()).hexdigest()=='2d3e52b871c842b28151dc0368dcce1036289d5d2b963f9dfdcd1a190bae62ae'
metadata={r['record_id']:r for r in csv.DictReader((B/'Classification/metadata/dataset_manifest.csv').open())};data={};archived={}
with zipfile.ZipFile(path) as z:
 for name in names:
  records=list(csv.DictReader(io.StringIO(z.read(name+'/evaluation/test_predictions.csv').decode())));data[name]={r['record_id']:r for r in records};archived[name]=json.loads(z.read(name+'/evaluation/test_metrics.json'))
ids=list(data[names[0]]);truth=np.array([C.index(data[names[0]][id]['true_class']) for id in ids]);assert len(ids)==3988;records=[metadata[id] for id in ids];assert all(r['training_role']=='test' for r in records);g=grouping(records);G=int(g.max())+1;rng=np.random.default_rng(2627);draws=rng.multinomial(G,np.ones(G)/G,size=2500);cm={};point={};boot={}
with threadpool_limits(limits=2):
 for name in names:
  assert set(ids)==set(data[name]);np.testing.assert_array_equal(truth,[C.index(data[name][id]['true_class']) for id in ids]);pred=np.array([C.index(data[name][id]['predicted_class']) for id in ids]);cm[name]=grouped_cm(truth,pred,g);point[name]=metrics(cm[name].sum(0));np.testing.assert_allclose(point[name]['balanced_accuracy'],archived[name]['balanced_accuracy'],atol=1e-14);boot[name]=metrics(np.tensordot(draws,cm[name],axes=(1,0)))
contrasts=[('original_baseline','fusion_refined','Primary: refined fusion − original'),('image_only','localized_no_morphology','Global + ROI − image only'),('fusion_zero_morphology','fusion_refined','Measured − zero morphology'),('ensemble_control','ensemble_anatomy','Anatomy − control ensemble')]
report={'scope':'Descriptive reanalysis of historically frozen TEST predictions, not a new confirmatory study or optimization target. No new image inference or training. Multiple-comparison adjustment applies only to these four contrasts, not earlier project search.','archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'helper_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'records':len(truth),'connected_image_groups':G,'bootstrap_draws':2500,'group_swap_permutations':5000,'seed':2627,'models':{},'contrasts':{}}
for name in names:report['models'][name]={'balanced_accuracy':float(point[name]['balanced_accuracy']),'macro_f1':float(point[name]['macro_f1']),'accuracy':float(point[name]['accuracy']),'balanced_accuracy_group_95_ci':ci(boot[name]['balanced_accuracy']),'macro_f1_group_95_ci':ci(boot[name]['macro_f1'])}
support=np.bincount(truth,minlength=4)
for a,b,label in contrasts:
 delta=float(point[b]['balanced_accuracy']-point[a]['balanced_accuracy']);v=((np.diagonal(cm[b],axis1=1,axis2=2)-np.diagonal(cm[a],axis1=1,axis2=2))/support).sum(1)/4;np.testing.assert_allclose(v.sum(),delta,atol=1e-14);null=(rng.integers(0,2,size=(5000,G))*2-1)@v
 report['contrasts'][label]={'from':a,'to':b,'balanced_accuracy_difference':delta,'paired_group_95_ci':ci(boot[b]['balanced_accuracy']-boot[a]['balanced_accuracy']),'group_swap_two_sided_p':float((1+(abs(null)>=abs(delta)-1e-14).sum())/5001),'macro_f1_difference':float(point[b]['macro_f1']-point[a]['macro_f1']),'macro_f1_paired_group_95_ci':ci(boot[b]['macro_f1']-boot[a]['macro_f1'])}
ordered=sorted(report['contrasts'],key=lambda k:report['contrasts'][k]['group_swap_two_sided_p']);running=0
for i,k in enumerate(ordered):running=max(running,min(1,(4-i)*report['contrasts'][k]['group_swap_two_sided_p']));report['contrasts'][k]['holm_adjusted_p_four_contrasts']=running
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,5.5))
for i,name in enumerate(names):
 p=100*float(point[name]['balanced_accuracy']);lo,hi=np.array(ci(boot[name]['balanced_accuracy']))*100;ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',capsize=4,color='#285A8E');ax.text(p+.2,i+.13,f'{p:.2f}%',fontsize=8)
ax.set(yticks=range(len(names)),yticklabels=labels,xlabel='Balanced accuracy (%)',title='Frozen historical TEST: model estimates and image-group bootstrap 95% intervals');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(F/('12_historical_model_comparison.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5.5))
for i,(label,d) in enumerate(report['contrasts'].items()):
 p=d['balanced_accuracy_difference']*100;lo,hi=np.array(d['paired_group_95_ci'])*100;ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',capsize=4);ax.text(hi+.05,i,f"Holm p={d['holm_adjusted_p_four_contrasts']:.3f}",va='center',fontsize=8)
ax.axvline(0,ls='--',color='grey');ax.set(yticks=range(4),yticklabels=list(report['contrasts']),xlabel='Balanced-accuracy difference (percentage points)',title='Descriptive historical TEST contrasts; paired whole-group resampling');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(F/('13_historical_paired_comparisons.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4)
for offset,name,label in [(-.16,'original_baseline','Original hybrid'),(0,'fusion_zero_morphology','Zero morphology'),(.16,'fusion_refined','Measured morphology')]:
 p=point[name]['recall'];lo,hi=np.array(ci(boot[name]['recall']));ax.errorbar(x+offset,p,yerr=[p-lo,hi-p],fmt='o',capsize=4,label=label)
ax.set(xticks=x,xticklabels=C,ylabel='Recall',ylim=(.6,1),title='Frozen TEST anatomy comparison: class recall and grouped 95% intervals');ax.legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(F/('14_historical_anatomy_class_recall.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
(O/'historical_project_statistics.json').write_text(json.dumps(report,indent=2));
with (O/'historical_model_metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','balanced_accuracy','BA_CI_low','BA_CI_high','macro_f1','accuracy'])
 for name,d in report['models'].items():w.writerow([name,d['balanced_accuracy'],*d['balanced_accuracy_group_95_ci'],d['macro_f1'],d['accuracy']])
print(json.dumps(report['contrasts'],indent=2))
