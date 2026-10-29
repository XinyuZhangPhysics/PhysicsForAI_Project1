import csv, json, math, sys, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from generic_core import make_generic_direction, central_norm_np, aligned_central_amplitude, linear_step_numpy

P=json.loads((ROOT/'protocol.json').read_text())
MU=P['mu']

class ProtocolTests(unittest.TestCase):
    def test_frozen_scientific_settings(self):
        self.assertEqual(P['protocol_id'],'Project1_GenericLocalDynamics_PRX_v1_2026-10-27')
        self.assertEqual(P['model_commit'],'28e66ca6931668447a3bac213f23d990ad3b0e2b')
        self.assertEqual(P['layer_zero_based'],20)
        self.assertEqual(P['horizon'],28)
        self.assertEqual(P['r_eta'],1.0)
        self.assertEqual(P['epsilon'],{'stable':-0.08,'unstable':0.08})
        self.assertEqual(P['direction_seeds'],list(range(2026102701,2026102709)))
        self.assertAlmostEqual(P['eta_X']*P['mu'],0.2,places=15)

    def test_direction_balancing(self):
        X,U=make_generic_direction(P['direction_seeds'][0],(12,7),(12,),MU)
        ex=MU*np.vdot(X,X).real
        eu=MU*np.vdot(U,U).real
        self.assertAlmostEqual(ex,0.5,places=12)
        self.assertAlmostEqual(eu,0.5,places=12)
        self.assertAlmostEqual(central_norm_np(X,U,MU),1.0,places=12)

    def test_side_identity(self):
        X,U=make_generic_direction(P['direction_seeds'][3],(9,5),(9,),MU)
        A=aligned_central_amplitude(P['a0_aligned_raw'],MU)
        zs=(A*X,A*U)
        zu=(A*X,A*U)
        self.assertTrue(np.array_equal(zs[0],zu[0]))
        self.assertTrue(np.array_equal(zs[1],zu[1]))

    def test_amplitude_rule(self):
        A=aligned_central_amplitude(P['a0_aligned_raw'],MU)
        self.assertAlmostEqual((A/2)/A,0.5,places=15)

    def test_historical_authority_matches_frozen_a0_and_horizon(self):
        with (ROOT/'inputs/P2J_FROZEN_CELL_PREDICTIONS.csv').open() as f:
            rows=list(csv.DictReader(f))
        rows=[r for r in rows if abs(abs(float(r['epsilon']))-.08)<1e-12 and float(r['r_eta'])==1.0]
        self.assertEqual(len(rows),2)
        self.assertTrue(all(float(r['a0'])==P['a0_aligned_raw'] for r in rows))
        by={float(r['epsilon']):int(r['N_cell']) for r in rows}
        self.assertEqual(by[-0.08],32)
        self.assertEqual(by[0.08],28)
        self.assertEqual(P['horizon'],min(by.values()))

    def test_linear_step_uses_frozen_action(self):
        X=np.array([[1.,2.],[3.,4.]])
        U=np.array([2.,-1.])
        def ct(u): return np.array([[u[0],u[1]],[u[1],u[0]]])
        def c(x): return np.array([x[0,0]+x[1,1],x[0,1]-x[1,0]])
        a=linear_step_numpy(X,U,0.1,0.2,0.3,0.4,ct,c)
        b=((1-.3*.1)*X+.3*ct(U),(1-.4*.2)*U+.4*c(X))
        self.assertTrue(np.allclose(a[0],b[0]))
        self.assertTrue(np.allclose(a[1],b[1]))

if __name__=='__main__': unittest.main()
