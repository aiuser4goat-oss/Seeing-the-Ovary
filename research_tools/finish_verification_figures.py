from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets/Accuracy_Development_2026_11_22');O=D/'normal_benign_specialist_v1/statistics';F=O/'figures'
def save(fig,path):
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=220,bbox_inches='tight')
 plt.close(fig)
for json_path,field,interval,title,path in [(O/'specialist_statistics.json','contrasts','paired_group_bootstrap_95_ci','Exploratory SELECT specialist effects',F/'04_paired_effects'),(O/'historical_project/historical_project_statistics.json','contrasts','paired_group_95_ci','Descriptive frozen TEST contrasts',O/'historical_project/figures/13_historical_paired_comparisons')]:
 data=json.loads(json_path.read_text())[field];fig,ax=plt.subplots(figsize=(11,5.5));labels=[]
 for i,(label,v) in enumerate(data.items()):
  p=v['balanced_accuracy_difference']*100;lo,hi=np.array(v[interval])*100;ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',capsize=4);adjusted=v['holm_adjusted_p_three_contrasts'] if 'holm_adjusted_p_three_contrasts' in v else v['holm_adjusted_p_four_contrasts'];ax.text(hi+.05,i,f'Holm p={adjusted:.3f}',va='center',fontsize=8);labels.append(label.replace('unchanged_main__to__plain_specialist','Plain − main').replace('unchanged_main__to__mined_specialist','Mined − main').replace('plain_specialist__to__mined_specialist','Mined − plain'))
 ax.axvline(0,ls='--',color='grey');ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Balanced-accuracy difference (percentage points)',title=title);fig.text(.5,-.015,'Intervals: individual unadjusted95%group-bootstrap CIs. P-values: Holm adjusted within this comparison family.',ha='center',fontsize=9);save(fig,path)
B=D.parent;test=list(csv.DictReader((B/'Preserved_Best_Completed_Version_2026_11_22/results/ensemble_control/evaluation/test_predictions.csv').open()));confidence=np.array([float(r['confidence']) for r in test]);correct=np.array([r['true_class']==r['predicted_class'] for r in test]);retained=np.array([r['referred']=='False' for r in test]);order=np.argsort(-confidence,kind='stable');coverage=np.arange(1,len(test)+1)/len(test);risk=np.cumsum(~correct[order])/np.arange(1,len(test)+1)
fig,ax=plt.subplots(figsize=(8,5));ax.plot(coverage,risk,label='Confidence ordering, descriptive curve');ax.scatter(retained.mean(),(~correct[retained]).mean(),color='#D17A22',zorder=3,label='Previously fixed calibration referral rule');ax.set(xlabel='Fraction of all records retained',ylabel='Error rate among retained records',title='Frozen TEST risk–coverage; subset accuracy is not full-set accuracy');ax.legend();fig.text(.5,-.01,f'Fixed rule retains{100*retained.mean():.2f}% of records; retained accuracy{100*correct[retained].mean():.2f}%. No new threshold chosen.',ha='center',fontsize=9);save(fig,F/'15_historical_risk_coverage')
d=json.loads((O/'historical_test_descriptive.json').read_text());values=d['bootstrap_variance_covariance_contributions'];fig,ax=plt.subplots(figsize=(8,5));ax.bar(values.keys(),np.array(list(values.values()))*100,color='#285A8E');ax.set(ylabel='Covariance allocation of bootstrap BA variance (%)',title='Frozen TEST uncertainty: covariance-based bootstrap decomposition');fig.text(.5,-.01,'Allocation includes cross-class covariance; not shares of mistakes or causal class attribution.',ha='center',fontsize=9);save(fig,F/'16_historical_variance_allocation')
print('Finished16figure pairs with interval/multiplicity labels and descriptive referral/variance graphs.')
