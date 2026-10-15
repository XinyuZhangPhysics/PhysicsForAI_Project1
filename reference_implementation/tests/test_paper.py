"""Independent mathematical tests; frozen-data checks run in reproduce_paper.py."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from moe_private_response import selection,mode_conversion,formation,native_tomography as native,q3,finite_time
class PaperChecks(unittest.TestCase):
    def test_selection_and_g1_expert_branch(self):
        for g in [1,2,4,8]:
            A,C,M=selection.projectors(4,g)
            self.assertTrue(np.allclose(A@C,0));self.assertTrue(np.allclose(A@M,0));self.assertTrue(np.allclose(A+C+M,np.eye(4*g)-np.ones((4*g,4*g))/(4*g)))
            q,k=selection.copied_fields(np.array([.1,.2,.3,.4]),np.array([.4,.3,.2,.1]),np.arange(g))
            self.assertTrue(np.allclose(A@q,q));self.assertTrue(np.allclose(C@k,k));self.assertAlmostEqual(selection.response(q,k),0)
        q,k=selection.copied_fields(np.array([.1,.2,.3,.4]),np.array([.1,.2,.3,.4]),[1.])
        self.assertGreater(selection.first_variation(q,k,np.zeros(4),q),0)
    def test_signed_mixed_pushforward(self):
        rng=np.random.default_rng(23);e=rng.normal(size=(4,3));e=e-e.mean(0)-e.mean(1)[:,None]+e.mean();ax=rng.normal(size=4);ay=rng.normal(size=4);ax-=ax.mean();ay-=ay.mean();s=1e-5
        derivative=(mode_conversion.child_pushforward(ax,ay,.7,-.2,e,s)-mode_conversion.child_pushforward(ax,ay,.7,-.2,e,-s))/(2*s)
        np.testing.assert_allclose(derivative,mode_conversion.mixed_slope(ax,ay,.7,-.2,e),rtol=1e-9,atol=1e-11)
    def test_cross_hessian_adjoint_and_growth(self):
        rng=np.random.default_rng(7);h=rng.normal(size=(7,3));phi=rng.normal(size=(7,4));grad=rng.normal(size=(7,2));op=formation.CrossOperator(h,phi,grad);m=rng.normal(size=(2,4));u=rng.normal(size=3)
        self.assertAlmostEqual(op.c(m)@u,np.sum(m*op.ct(u)),places=12)
        mu=.3;kappa=.2;lam=.1;em=.04;eu=.07;L=finite_time.local_map(np.array([[kappa]]),mu,lam,em,eu)
        self.assertAlmostEqual(formation.growth_rate(mu,kappa,lam,em,eu),np.log(np.linalg.eigvals(L).max()),places=13)
    def test_native_tangent_and_metric(self):
        p=np.array([[.1,.2,.3,.4]]);idx=np.array([[2,3]]);out=np.array([[[1.,2.],[3.,4.]]]);gain=np.array([.1,.2,.3,.4]);bias=np.array([-.2,.3,.5,-.6]);step=1e-6
        def f(t):
            logits=np.log(p)+t*bias;probs=mode_conversion.softmax(logits);return np.einsum('tk,tkh->th',probs[:,[2,3]]*(1+t*gain[[2,3]]),out)
        np.testing.assert_allclose((f(step)-f(-step))/(2*step),native.soft_tangent(p,idx,out,gain,bias),rtol=1e-8)
        B=np.array([[2.,.2],[.2,1.]]);H=np.array([[3.,.1],[.1,2.]]);S=np.array([[2.,.3],[0.,.7]])
        np.testing.assert_allclose(np.linalg.eigvalsh(native.physical_operator(B,H)),np.linalg.eigvalsh(native.physical_operator(S.T@B@S,S.T@H@S)),rtol=1e-12)
        self.assertTrue(native.scalar_gate(B,3*B,3)['adequate']);self.assertFalse(native.split_gate(B,np.diag([.01,20]))['stable'])
    def test_q3_zero_and_shape(self):
        x=np.arange(128,dtype=np.float32).reshape(2,64);b=np.linspace(-1,1,64);np.testing.assert_array_equal(q3.logits_action(x,b,0),x)
        self.assertEqual(q3.logits_action(x,b,.1).dtype,np.float32)
        with self.assertRaises(ValueError):q3.logits_action(x,np.ones(63),1)
    def test_finite_order_and_cumulant(self):
        C=[np.array([[.02,.01]]),np.array([[-.01,.03]]),np.array([[.01,-.02]])];maps=[finite_time.local_map(c,.1,.2,.01,.03) for c in C];v=finite_time.closures(maps,maps)
        np.testing.assert_allclose(v['Phi_ord'],maps[2]@maps[1]@maps[0],rtol=1e-13)
        self.assertLess(np.linalg.norm(v['Phi_log+comm']-v['Phi_ord']),np.linalg.norm(v['Phi_log']-v['Phi_ord']))
        same=finite_time.closures([maps[0]]*3,[maps[0]])
        np.testing.assert_allclose(same['Phi_log+comm'],same['Phi_ord'],atol=1e-13)
if __name__=='__main__':unittest.main()
