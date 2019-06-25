import itertools
import numpy
import os
import unittest
from pyscf import gto, ao2mo, scf
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.from_pyscf import integrals_from_scf, integrals_from_chkfile
from pauxy.utils.misc import dotdict
from pauxy.walkers.multi_det import MultiDetWalker

class TestMultiDetWalker(unittest.TestCase):

    def test_walker_overlap(self):
        system = dotdict({'nup': 5, 'ndown': 5, 'nbasis': 10,
                          'nelec': (5,5), 'ne': 10})
        numpy.random.seed(7)
        a = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        wfn = (a + 1j*b).reshape((3,system.nbasis,system.nup+system.ndown))
        coeffs = numpy.array([0.5+0j,0.3+0j,0.1+0j])
        trial = MultiSlater(system, (coeffs, wfn))
        walker = MultiDetWalker({}, system, trial)
        def calc_ovlp(a,b):
            return numpy.linalg.det(numpy.dot(a.conj().T, b))
        ovlp = 0.0+0j
        na = system.nup
        pa = trial.psi[0,:,:na]
        pb = trial.psi[0,:,na:]
        for i, d in enumerate(trial.psi):
            ovlp += coeffs[i].conj()*calc_ovlp(d[:,:na],pa)*calc_ovlp(d[:,na:],pb)
        self.assertAlmostEqual(ovlp.real,walker.ovlp.real)
        self.assertAlmostEqual(ovlp.imag,walker.ovlp.imag)
        # Test PH type wavefunction.
        orbs = numpy.arange(system.nbasis)
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        oa, ob = zip(*itertools.product(oa,ob))
        oa = oa[:5]
        ob = ob[:5]
        coeffs = numpy.array([0.9, 0.01, 0.01, 0.02, 0.04],
                             dtype=numpy.complex128)
        wfn = (coeffs,oa,ob)
        a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        init = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
        trial = MultiSlater(system, wfn, init=init)
        walker = MultiDetWalker({}, system, trial)
        I = numpy.eye(system.nbasis)
        ovlp_sum = 0.0
        for idet, (c, occa, occb) in enumerate(zip(coeffs,oa,ob)):
            psia = I[:,occa]
            psib = I[:,occb]
            sa = numpy.dot(psia.conj().T, init[:,:system.nup])
            sb = numpy.dot(psib.conj().T, init[:,system.nup:])
            ga = numpy.dot(init[:,:system.nup], numpy.dot(sa, psia.conj().T)).T
            gb = numpy.dot(init[:,system.nup:], numpy.dot(sb, psib.conj().T)).T
            ovlp *= c.conj()*numpy.linalg.det(sa)*numpy.linalg.det(sb)
            ovlp_sum += ovlp
            walk_ovlp = numpy.linalg.det(walker.inv_ovlp[0][idet])*numpy.linalg.det(walker.inv_ovlp[1][idet])
            print(ovlp,walk_ovlp)
            self.assertAlmostEqual(ovlp, walk_ovlp)
            self.assertTrue(numpy.linalg.norm(ga-walker.Gi[idet,0]) < 1e-8)
            self.assertTrue(numpy.linalg.norm(gb-walker.Gi[idet,1]) < 1e-8)
        self.assertAlmostEqual(ovlp_sum, walker.ovlp)
