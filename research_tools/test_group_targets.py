import sys,unittest
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0,'/Users/ik-mac/ovarian_ai_project copy')
from accuracy_development.group_targets import aggregate_targets,objective_gradient
class GroupTests(unittest.TestCase):
 def setUp(self):
  self.a=aggregate_targets(['a','a','b','c','c'],np.array([0,1,2,1,1]),[.4,1.3,2.,.5,.8])
  self.z=np.array([[1.,-.3,.4,0],[.2,.3,1.4,.7],[.8,.1,-.2,.4]])
 def test_record_objective_equivalence(self):
  loss,_=objective_gradient(self.z,self.a['soft_targets'],self.a['group_weights'],'soft_ce')
  z=self.z[self.a['record_to_group']]; y=np.array([0,1,2,1,1]);w=np.array([.4,1.3,2,.5,.8])
  expected=np.mean(w*(logsumexp(z,axis=1)-z[np.arange(5),y]))
  self.assertAlmostEqual(loss,expected,places=12)
  self.assertEqual(self.a['records_represented'],5)
 def test_gradients(self):
  for kind,targets in [('soft_ce',self.a['soft_targets']),('candidate_marginal',self.a['candidate_mask'])]:
   _,grad=objective_gradient(self.z,targets,self.a['group_weights'],kind)
   for i in range(3):
    for j in range(4):
     plus=self.z.copy();minus=self.z.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
     numeric=(objective_gradient(plus,targets,self.a['group_weights'],kind)[0]-objective_gradient(minus,targets,self.a['group_weights'],kind)[0])/2e-6
     self.assertAlmostEqual(grad[i,j],numeric,places=7)
 def test_singleton_equivalence(self):
  q=np.eye(4)[[0,2,1]];w=np.ones(3)
  a=objective_gradient(self.z,q,w,'soft_ce');b=objective_gradient(self.z,q.astype(bool),w,'candidate_marginal')
  self.assertAlmostEqual(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])
 def test_role_guard(self):
  for role in ['select','calibrate','test']:
   with self.assertRaises(ValueError):aggregate_targets(['a'],np.array([0]),[1],role=role)
 def test_invalid_weights(self):
  for w in [[0],[np.nan],[-1]]:
   with self.assertRaises(ValueError):aggregate_targets(['a'],np.array([0]),w)
if __name__=='__main__':unittest.main()
