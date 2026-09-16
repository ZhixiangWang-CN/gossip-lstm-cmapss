import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from pdm.data import load_raw,prepare,windows
from pdm.model import LSTM,vector,put_vector
from pdm.network import mixing_matrix,edges_for,communicate
from pdm.metrics import from_counts,metrics
from pdm.suites import DEFAULT,suite

ROOT=Path(__file__).resolve().parents[1]

class ProtocolTests(unittest.TestCase):
    def test_data_split_and_scaler(self):
        c=copy.deepcopy(DEFAULT);train,val,test,a=prepare(ROOT/'data',c)
        self.assertFalse(set(train.meta.engine)&set(val.meta.engine))
        self.assertEqual(len(test.meta),8255)
        self.assertEqual(int(test.meta.y.sum()),332)
        self.assertEqual(int(test.meta.terminal.sum()),93)
        self.assertEqual(len(train.meta)+len(val.meta),15731)
        self.assertEqual(a['features'][3],'cycle_norm')
        self.assertNotIn('RUL',a['features'])

    def test_endpoint_not_next_cycle(self):
        raw=np.zeros((55,26));raw[:,0]=1;raw[:,1]=np.arange(1,56)
        w=windows(raw,[1],np.zeros(25),np.ones(25),length=50,horizon=5)
        self.assertEqual(len(w.meta),6)
        self.assertEqual(w.meta.iloc[0].cycle,50)
        self.assertEqual(w.meta.iloc[0].rul,5)
        self.assertEqual(w.x[0,-1,3],50)

    def test_payload_and_round(self):
        torch.set_num_threads(2)
        m=LSTM();self.assertEqual(vector(m).numel(),80651)
        c=copy.deepcopy(DEFAULT);vs=[torch.full((80651,),float(i)) for i in range(10)]
        new,_,_,ledger,_=communicate(vs,vs[0],[10]*10,c,0,None)
        self.assertEqual(sum(r['bytes'] for r in ledger),80651*4*20)
        self.assertTrue(torch.allclose(new[0],torch.full_like(vs[0],10/3)))
        put_vector(m,new[0]);self.assertTrue(torch.equal(vector(m),new[0]))
        c['method']='fedavg'
        new,server,_,ledger,_=communicate(vs,vs[0],[10]*10,c,0,None)
        self.assertTrue(torch.allclose(server,torch.full_like(server,4.5)))
        self.assertEqual(sum(r['bytes'] for r in ledger),80651*4*20)

    def test_mixing_conservation_and_disconnect(self):
        for n in [3,5,10,40]:
            w=mixing_matrix(n,edges_for(n,'ring'))
            np.testing.assert_allclose(w.sum(0),1)
            np.testing.assert_allclose(w.sum(1),1)
            self.assertGreaterEqual(w.min(),0)
        w=mixing_matrix(4,[(0,1)])
        np.testing.assert_array_equal(w[3],[0,0,0,1])

    def test_event_suppression_includes_control(self):
        c=copy.deepcopy(DEFAULT);c['method']='event_gossip';c['nodes']=3
        vs=[torch.ones(10) for _ in range(3)]
        new,sv,st,first,_=communicate(vs,vs[0],[1]*3,c,0,None)
        new,_,_,second,_=communicate(new,sv,[1]*3,c,1,st)
        self.assertTrue(any(r['kind']=='model' for r in first))
        self.assertFalse(any(r['kind']=='model' for r in second))
        self.assertEqual(sum(r['bytes'] for r in second),6*16)

    def test_pooled_metrics(self):
        m=from_counts(7749,81,93,239)
        self.assertAlmostEqual(m['f1'],.7331288343558282)
        self.assertAlmostEqual(m['recall'],239/332)
        self.assertNotAlmostEqual(m['recall'],.801)

    def test_sample_mass_stationarity(self):
        mass=np.array([1.,2.,4.,8.]);w=mixing_matrix(4,edges_for(4,'ring'),mass)
        np.testing.assert_allclose(w.sum(1),1)
        np.testing.assert_allclose(mass@w,mass)
        self.assertGreaterEqual(w.min(),0)

    def test_report_keeps_protocols_separate(self):
        from pdm.report import summary
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as d, patch('pdm.report.savefig'):
            root=Path(d)
            jobs=[('gossip',11,1),('gossip',22,1),('fedavg',11,1),('fedavg',22,1),('gossip',11,2)]
            for i,(method,seed,epochs) in enumerate(jobs):
                p=root/str(i);p.mkdir()
                c={**DEFAULT,'method':method,'seed':seed,'epochs':epochs}
                r=dict(config=c,metrics={'test_terminal/fixed':metrics([0,0,1,1],[.1,.2,.8,.9])},bytes_total=100,
                       training_wall_s=1,modeled_parallel_s=1)
                (p/'results.json').write_text(json.dumps(r));(p/'DONE.json').write_text('{}')
            summary(root)
            a=pd.read_csv(root/'seed_summary.csv');self.assertEqual(sorted(a.runs.tolist()),[1,2,2])
            b=pd.read_csv(root/'paired_tests.csv');self.assertEqual(b.n_pairs.tolist(),[2])
            self.assertEqual(b.exact_signflip_p.tolist(),[1.])
            plt.close('all')

    def test_resume_equals_uninterrupted(self):
        from pdm import engine
        c=suite('smoke')[3];c['epochs']=2;c['bootstrap']=10
        with tempfile.TemporaryDirectory() as d, patch('pdm.report.job_plots'):
            a=Path(d)/'a';b=Path(d)/'b'
            engine.run_job(c,ROOT/'data',a)
            real_replace=engine.os.replace;interrupt=[True]
            def stop_once(src,dest):
                real_replace(src,dest)
                if str(dest).endswith('checkpoint.pt') and interrupt[0]:
                    interrupt[0]=False;raise KeyboardInterrupt()
            with patch('pdm.engine.os.replace',side_effect=stop_once):
                with self.assertRaises(KeyboardInterrupt):engine.run_job(c,ROOT/'data',b)
            engine.run_job(c,ROOT/'data',b)
            x=pd.read_csv(a/'predictions_test.csv.gz');y=pd.read_csv(b/'predictions_test.csv.gz')
            np.testing.assert_array_equal(x.prob,y.prob)
            self.assertTrue(json.loads((b/'checks.json').read_text())['passed'])

if __name__=='__main__':unittest.main()
