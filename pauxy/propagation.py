"""Routines for performing propagation of a walker"""

import numpy
import scipy.linalg
import math
import cmath
import copy
import pauxy.utils


def get_propagator(options, qmc, system, trial):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pauxy.qmc.QMCOpts` class
        Trial wavefunction input options.
    system : class
        System class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    hs_type = options.get('hubbard_stratonovich', 'discrete')
    if hs_type == 'discrete':
        propagator = DiscreteHubbard(options, qmc, system, trial)
    elif hs_type == "hubbard_continuous":
        propagator = ContinuousHubbard(options, qmc, system, trial)
    elif hs_type == "continuous":
        propagator = GenericContinuous(options, qmc, system, trial)
    else:
        propagator = None

    return propagator


def local_energy_bound(local_energy, mean, threshold):
    """Try to suppress rare population events by imposing local energy bound.

    See: Purwanto et al., Phys. Rev. B 80, 214116 (2009).

Parameters
----------
local_energy : float
    Local energy of current walker
mean : float
    Mean value of local energy about which we impose the threshold / bound.
threshold : float
    Amount of lee-way for energy fluctuations about the mean.
"""

    maximum = mean + threshold
    minimum = mean - threshold

    if (local_energy >= maximum):
        local_energy = maximum
    elif (local_energy < minimum):
        local_energy = minimum
    else:
        local_energy = local_energy

    return local_energy


def calculate_overlap_ratio_multi_ghf(walker, delta, trial, i):
    nbasis = trial.psi.shape[1] // 2
    for (idx, G) in enumerate(walker.Gi):
        guu = G[i,i]
        gdd = G[i+nbasis,i+nbasis]
        gud = G[i,i+nbasis]
        gdu = G[i+nbasis,i]
        walker.R[idx,0] = (
            (1+delta[0,0]*guu)*(1+delta[0,1]*gdd) - delta[0,0]*gud*delta[0,1]*gdu
        )
        walker.R[idx,1] = (
            (1+delta[1,0]*guu)*(1+delta[1,1]*gdd) - delta[1,0]*gud*delta[1,1]*gdu
        )
    R = numpy.einsum('i,ij,i->j',trial.coeffs,walker.R,walker.ots)/walker.ot
    return 0.5 * numpy.array([R[0],R[1]])

def calculate_overlap_ratio_multi_det(walker, delta, trial, i):
    for (idx, G) in enumerate(walker.Gi):
        walker.R[idx,0,0] = (1+delta[0][0]*G[0][i,i])
        walker.R[idx,0,1] = (1+delta[0][1]*G[1][i,i])
        walker.R[idx,1,0] = (1+delta[1][0]*G[0][i,i])
        walker.R[idx,1,1] = (1+delta[1][1]*G[1][i,i])
    spin_prod = numpy.einsum('ikj,ji->ikj',walker.R,walker.ots)
    R = numpy.einsum('i,ij->j',trial.coeffs,spin_prod[:,:,0]*spin_prod[:,:,1])/walker.ot
    return 0.5 * numpy.array([R[0],R[1]])

def calculate_overlap_ratio_single_det(walker, delta, trial, i):
    R1 = (1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i])
    R2 = (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])
    return 0.5 * numpy.array([R1,R2])

def generic_continuous(walker, state):
    r"""Continuous HS transformation

    This form assumes nothing about the form of the two-body Hamiltonian and
    is thus quite slow, particularly if the matrix is M^2xM^2.

    Todo: check if this actually works.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
        updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """

    # iterate over spins

    dt = state.dt
    gamma = state.system.gamma
    nup = state.system.nup
    # Generate ~M^2 normally distributed auxiliary fields.
    sigma = dt**0.5 * numpy.random.normal(0.0, 1.0, len(gamma))
    # Construct HS potential, V_HS = sigma dot U
    V_HS = numpy.einsum('ij,j->i', sigma, gamma)
    # Reshape so we can apply to MxN Slater determinant.
    V_HS = numpy.reshape(V_HS, (M, M))
    for n in range(1, nmax_exp + 1):
        EXP_V = EXP_V + numpy.dot(V_HS, EXP_V) / math.factorial(n)
    walker.phi[:nup] = numpy.dot(EXP_V, walker.phi[:nup])
    walker.phi[nup:] = numpy.dot(EXP_V, walker.phi[:nup])

    # Update inverse and green's function
    walker.inverse_overlap(trial)
    walker.greens_function(trial)
    # Perform importance sampling, phaseless and real local energy
    # approximation and update
    E_L = state.estimators.local_energy(system, walker.G)[0].real
    ot_new = walker.calc_otrial(trial)
    dtheta = cmath.phase(ot_new/walker.ot)
    walker.weight = (walker.weight * exp(-0.5*system.dt*(walker.E_L-E_L))
                                  * max(0, cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def kinetic_importance_sampling(walker, state):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    self.propagators.kinetic(walker.phi, state)
    # Update inverse overlap
    walker.inverse_overlap(state.trial.psi)
    # Update walker weight
    ot_new = walker.calc_otrial(state.trial)
    ratio = (ot_new/walker.ot)
    phase = cmath.phase(ratio)
    if abs(phase) < math.pi/2:
        walker.weight = walker.weight * ratio.real
        walker.ot = ot_new
        # Todo : remove computation of green's function repeatedly.
        walker.greens_function(state.trial)
    else:
        walker.weight = 0.0


def kinetic_real(phi, system, bt2):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    nup = system.nup
    # Assuming that our walker is in UHF form.
    phi[:,:nup] = bt2[0].dot(phi[:,:nup])
    phi[:,nup:] = bt2[1].dot(phi[:,nup:])


def kinetic_ghf(phi, system, bt2):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the GHF algorithm.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    nup = system.nup
    nb = system.nbasis
    # Assuming that our walker is in GHF form.
    phi[:nb,:nup] = bt2.dot(phi[:nb,:nup])
    phi[nb:,nup:] = bt2.dot(phi[nb:,nup:])


def propagate_potential_auxf(phi, state, field_config):
    """Propagate walker given a fixed set of auxiliary fields.

    Useful for debugging.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        Walker's slater determinant to be updated.
    state : :class:`pauxy.state.State`
        Simulation state.
    field_config : numpy array
        Auxiliary field configurations to apply to walker.
    """

    bv_up = numpy.array([state.auxf[xi, 0] for xi in field_config])
    bv_down = numpy.array([state.auxf[xi, 1] for xi in field_config])
    phi[:,:nup] = numpy.einsum('i,ij->ij', bv_up, phi[:,:nup])
    phi[:,nup:] = numpy.einsum('i,ij->ij', bv_down, phi[:,nup:])

def construct_propagator_matrix(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    Parameters
    ----------
    config : numpy array
        Auxiliary field configuration.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    Bup = BT2[0].dot(bv_up).dot(BT2[0])
    Bdown = BT2[1].dot(bv_down).dot(BT2[1])

    if conjt:
        return numpy.array([Bup.conj().T, Bdown.conj().T])
    else:
        return numpy.array([Bup, Bdown])


def construct_propagator_matrix_generic(system, BT2, config, dt, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    Parameters
    ----------
    config : numpy array
        Auxiliary field configuration.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    VHS = 1j*dt**0.5*numpy.einsum('l,lpq->pq', config, system.chol_vecs)
    EXP_VHS = pauxy.utils.exponentiate_matrix(VHS)
    Bup = BT2[0].dot(EXP_VHS).dot(BT2[0])
    Bdown = BT2[1].dot(EXP_VHS).dot(BT2[1])

    if conjt:
        return [Bup.conj().T, Bdown.conj().T]
    else:
        return [Bup, Bdown]


def construct_propagator_matrix_ghf(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    Parameters
    ----------
    config : numpy array
        Auxiliary field configuration.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    BV = scipy.linalg.block_diag(bv_up, bv_down)
    B = BT2.dot(BV).dot(BT2)

    if conjt:
        return B.conj().T
    else:
        return B


def back_propagate(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for UHF style wavefunction.

    todo: Explanation.

    parameters
    ---------
    state : :class:`pauxy.state.state`
        state object
    psi_n : list of :class:`pauxy.walker.walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. on
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an itcf.
    step : int
        simulation step (modulo total number of fields to save). this is
        necessary when estimating an itcf for imaginary times >> back
        propagation time.

    returns
    -------
    psi_bp : list of :class:`pauxy.walker.walker` objects
        back propagated list of walkers.
    """

    psi_bp = [pauxy.walker.Walker(1, system, trial, w) for w in range(len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix(system, BT2,
                                            c, conjt=True)
            psi_bp[iw].phi[:,:nup] = B[0].dot(psi_bp[iw].phi[:,:nup])
            psi_bp[iw].phi[:,nup:] = B[1].dot(psi_bp[iw].phi[:,nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp

def back_propagate_generic(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for UHF style wavefunction.

    todo: Explanation.

    parameters
    ---------
    state : :class:`pauxy.state.state`
        state object
    psi_n : list of :class:`pauxy.walker.walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. on
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an itcf.
    step : int
        simulation step (modulo total number of fields to save). this is
        necessary when estimating an itcf for imaginary times >> back
        propagation time.

    returns
    -------
    psi_bp : list of :class:`pauxy.walker.walker` objects
        back propagated list of walkers.
    """

    psi_bp = [
        pauxy.walker.Walker(
            1,
            system,
            trial,
            w) for w in range(
            len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix_generic(system, BT2,
                                                    c, dt, conjt=True)
            psi_bp[iw].phi[:, :nup] = B[0].dot(psi_bp[iw].phi[:, :nup])
            psi_bp[iw].phi[:, nup:] = B[1].dot(psi_bp[iw].phi[:, nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp


def back_propagate_ghf(system, psi, trial, nstblz, BT2, dt):
    r"""perform backpropagation.

    todo: explanation and disentangle measurement from act.

    parameters
    ---------
    state : :class:`pauxy.state.State`
        state object
    psi_n : list of :class:`pauxy.walker.Walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. on
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an itcf.
    step : int
        simulation step (modulo total number of fields to save). this is
        necessary when estimating an itcf for imaginary times >> back
        propagation time.

    returns
    -------
    psi_bp : list of :class:`pauxy.walker.Walker` objects
        back propagated list of walkers.
    """

    psi_bp = [pauxy.walker.MultiGHFWalker(1, system, trial, w, weights='ones',
                                          wfn0='GHF') for w in range(len(psi))]
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix_ghf(system, BT2,
                                                c, conjt=True)
            for (idet, psi_i) in enumerate(psi_bp[iw].phi):
                # propagate each component of multi-determinant expansion
                psi_bp[iw].phi[idet] = B.dot(psi_bp[iw].phi[idet])
                if i != 0 and i % nstblz == 0:
                    # implicitly propagating the full GHF wavefunction
                    (psi_bp[iw].phi[idet], detR) = pauxy.utils.reortho(psi_i)
                    psi_bp[iw].weights[idet] *= detR.conjugate()
    return psi_bp


def back_propagate_single(phi_in, configs, weights,
                          system, nstblz, BT2, store=False):
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix(system, BT2, c, conjt=True)
        phi_in[:,:nup] = B[0].dot(phi_in[:,:nup])
        phi_in[:,nup:] = B[1].dot(phi_in[:,nup:])
        if i != 0 and i % nstblz == 0:
            (phi_in[:,:nup], R) = pauxy.utils.reortho(phi_in[:,:nup])
            (phi_in[:,nup:], R) = pauxy.utils.reortho(phi_in[:,nup:])
        if store:
            psi_store.append(copy.deepcopy(phi_in))

    return psi_store


def back_propagate_single_ghf(
        phi, configs, weights, system, nstblz, BT2, store=False):
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix_ghf(system, BT2, c, conjt=True)
        for (idet, psi_i) in enumerate(phi):
            # propagate each component of multi-determinant expansion
            phi[idet] = B.dot(phi[idet])
            if i != 0 and i % nstblz == 0:
                # implicitly propagating the full GHF wavefunction
                (phi[idet], detR) = pauxy.utils.reortho(psi_i)
                weights[idet] *= detR.conjugate()
        if store:
            psi_store.append(copy.deepcopy(phi))

    return psi_store


def propagate_single(psi, system, B):
    r"""Perform backpropagation for single configuration.

    explanation...

    Parameters
    ---------
    state : :class:`pauxy.state.State`
        state object
    psi : :class:`numpy.ndarray`
        Input wavefunction to propagate.
    B : numpy array
        Propagation matrix.
    """
    nup = system.nup
    if len(B.shape) == 3:
        psi[:,:nup] = B[0].dot(psi[:,:nup])
        psi[:,nup:] = B[1].dot(psi[:,nup:])
    else:
        M = system.nbasis
        psi[:M,:nup] = B[:M,:M].dot(psi[:M,:nup])
        psi[M:,nup:] = B[M:,M:].dot(psi[M:,nup:])


def kinetic_kspace(psi, system, btk):
    """Apply the kinetic energy projector in kspace.

    May be faster for very large dilute lattices.
    """
    s = system
    # Transform psi to kspace by fft-ing its columns.
    tup = pauxy.utils.fft_wavefunction(psi[:,:s.nup], s.nx, s.ny,
                                         s.nup, psi[:,:s.nup].shape)
    tdown = pauxy.utils.fft_wavefunction(psi[:,s.nup:], s.nx, s.ny,
                                           s.ndown, psi[:,s.nup:].shape)
    # Kinetic enery operator is diagonal in momentum space.
    # Note that multiplying by diagonal btk in this way is faster than using
    # einsum and way faster than using dot using an actual diagonal matrix.
    tup = (btk*tup.T).T
    tdown = (btk*tdown.T).T
    # Transform psi to kspace by fft-ing its columns.
    tup = pauxy.utils.ifft_wavefunction(tup, s.nx, s.ny, s.nup, tup.shape)
    tdown = pauxy.utils.ifft_wavefunction(tdown, s.nx, s.ny, s.ndown, tdown.shape)
    if psi.dtype == float:
        psi[:,:s.nup] = tup.astype(float)
        psi[:,s.nup:] = tdown.astype(float)
    else:
        psi[:,:s.nup] = tup
        psi[:,s.nup:] = tdown

class DiscreteHubbard:

    def __init__(self, options, qmc, system, trial):

        if trial.type == 'GHF':
            self.bt2 = scipy.linalg.expm(-0.5*qmc.dt*system.T[0])
        else:
            self.bt2 = numpy.array([scipy.linalg.expm(-0.5*qmc.dt*system.T[0]),
                                    scipy.linalg.expm(-0.5*qmc.dt*system.T[1])])
        if trial.type == 'GHF' and trial.bp_wfn is not None:
            self.BT_BP = scipy.linalg.block_diag(self.bt2, self.bt2)
            self.back_propagate = back_propagate_ghf
        else:
            self.BT_BP = self.bt2
            self.back_propagate = back_propagate
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5*qmc.dt*system.eks)
        self.hs_type = 'discrete'
        self.free_projection = options.get('free_projection', False)
        self.gamma = numpy.arccosh(numpy.exp(0.5*qmc.dt*system.U))
        self.auxf = numpy.array([[numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                                [numpy.exp(-self.gamma), numpy.exp(self.gamma)]])
        self.auxf = self.auxf * numpy.exp(-0.5*qmc.dt*system.U)
        self.delta = self.auxf - 1
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_constrained
        if trial.name == 'multi_determinant':
            if trial.type == 'GHF':
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_ghf
                self.kinetic = kinetic_ghf
                self.update_greens_function = self.update_greens_function_ghf
            else:
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_det
                self.kinetic = kinetic_real
        else:
            self.calculate_overlap_ratio = calculate_overlap_ratio_single_det
            self.update_greens_function = self.update_greens_function_uhf
            if qmc.ffts:
                self.kinetic = kinetic_kspace
            else:
                self.kinetic = kinetic_real

    def update_greens_function_uhf(self, walker, trial, i, nup):
        vup = trial.psi.conj()[i,:nup]
        uup = walker.phi[i,:nup]
        q = numpy.dot(walker.inv_ovlp[0], vup)
        walker.G[0][i,i] = numpy.dot(uup, q)
        vdown = trial.psi.conj()[i,nup:]
        udown = walker.phi[i,nup:]
        q = numpy.dot(walker.inv_ovlp[1], vdown)
        walker.G[1][i,i] = numpy.dot(udown, q)

    def update_greens_function_ghf(self, walker, trial, i, nup):
        walker.greens_function(trial)

    def kinetic_importance_sampling(self, walker, system, trial):
        r"""Propagate by the kinetic term by direct matrix multiplication.

        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
            appropriately.  updates inplace.
        state : :class:`pauxy.state.State`
            Simulation state.
        """
        self.kinetic(walker.phi, system, self.bt2)
        # Update inverse overlap
        walker.inverse_overlap(trial.psi)
        # Update walker weight
        ot_new = walker.calc_otrial(trial)
        ratio = (ot_new/walker.ot)
        phase = cmath.phase(ratio)
        if abs(phase) < 0.5*math.pi:
            walker.weight = walker.weight * ratio.real
            walker.ot = ot_new
        else:
            walker.weight = 0.0

    def two_body(self, walker, system, trial):
        r"""Propagate by potential term using discrete HS transform.

        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`b_V` and updated the weight appropriately.
            updates inplace.
        state : :class:`pauxy.state.State`
            Simulation state.
        """
        # Construct random auxilliary field.
        delta = self.delta
        nup = system.nup
        soffset = walker.phi.shape[0] - system.nbasis
        for i in range(0, system.nbasis):
            self.update_greens_function(walker, trial, i, nup)
            # Ratio of determinants for the two choices of auxilliary fields
            probs = self.calculate_overlap_ratio(walker, delta, trial, i)
            # issues here with complex numbers?
            phaseless_ratio = numpy.maximum(probs.real, [0,0])
            norm = sum(phaseless_ratio)
            r = numpy.random.random()
            # Is this necessary?
            # todo : mirror correction
            if norm > 0:
                walker.weight = walker.weight * norm
                if r < phaseless_ratio[0]/norm:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i,:nup] * delta[xi, 0]
                vtdown = walker.phi[i+soffset,nup:] * delta[xi, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i+soffset,nup:] = walker.phi[i+soffset,nup:] + vtdown
                walker.update_overlap(probs, xi, trial.coeffs)
                walker.field_configs.push(xi)
                walker.update_inverse_overlap(trial, vtup, vtdown, i)
            else:
                walker.weight = 0
                return

    def propagate_walker_constrained(self, walker, system, trial):
        r"""Wrapper function for propagation using discrete transformation

        The discrete transformation allows us to split the application of the
        projector up a bit more, which allows up to make use of fast matrix update
        routines since only a row might change.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """

        if abs(walker.weight) > 0:
            self.kinetic_importance_sampling(walker, system, trial)
        if abs(walker.weight) > 0:
            self.two_body(walker, system, trial)
        if abs(walker.weight.real) > 0:
            self.kinetic_importance_sampling(walker, system, trial)

    def propagate_walker_multi_site(self, walker, system, trial):
        r"""Wrapper function for propagation using discrete transformation

        The discrete transformation allows us to split the application of the
        projector up a bit more, which allows up to make use of fast matrix update
        routines since only a row might change.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """

        # 1. Apply kinetic projector.
        self.kinetic(walker.phi, system, self.bt2)
        # 2. Apply potential projector.
        propagate_potential_auxf(walker, state)
        # 3. Apply kinetic projector.
        self.kinetic(walker.phi, state)
        walker.inverse_overlap(trial.psi)
        # Calculate new total overlap and update components of overlap
        ot_new = walker.calc_otrial(trial.psi)
        # Now apply phaseless approximation
        dtheta = cmath.phase(ot_new/walker.ot)
        walker.weight = walker.weight * max(0, math.cos(dtheta))
        walker.ot = ot_new

    def propagate_walker_free(self, walker, system, trial):
        r"""Propagate walker without imposing constraint.

        Uses single-site updates for potential term.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        kinetic_real(walker.phi, system, self.bt2)
        delta = self.delta
        nup = system.nup
        for i in range(0, system.nbasis):
            if abs(walker.weight) > 0:
                r = numpy.random.random()
                # TODO: remove code repition.
                if r < 0.5:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i,:nup] * delta[xi, 0]
                vtdown = walker.phi[i,nup:] * delta[xi, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i,nup:] = walker.phi[i,nup:] + vtdown
        kinetic_real(walker.phi, system, self.bt2)
        walker.inverse_overlap(trial.psi)
        # Update walker weight
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)

class ContinuousHubbard:
    '''Base propagator class'''

    def __init__(self, options, qmc, system, trial):
        self.hs_type = 'hubbard_continuous'
        self.free_projection = options.get('free_projection', False)
        self.bt2 = numpy.array([scipy.linalg.expm(-0.5*qmc.dt*system.T[0]),
                                scipy.linalg.expm(-0.5*qmc.dt*system.T[1])])
        self.BT_BP = self.bt2
        self.back_propagate = back_propagate
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5*qmc.dt*system.eks)
        model = system.__class__.__name__
        self.dt = qmc.dt
        # optimal mean-field shift for the hubbard model
        self.mf_shift = (system.nup+system.ndown) / float(system.nbasis)
        self.iut_fac = 1j*numpy.sqrt((system.U*self.dt))
        self.ut_fac = self.dt*system.U
        # Include factor of M! bad name
        self.mf_nsq = system.nbasis * self.mf_shift**2.0
        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free_continuous
        else:
            self.propagate_walker = self.propagate_walker_constrained_continuous
        if qmc.ffts:
            self.kinetic = kinetic_kspace
        else:
            self.kinetic = kinetic_real

    def two_body(self, walker, system, trial):
        r"""Continuous Hubbard-Statonovich transformation for Hubbard model.

        Only requires M auxiliary fields.

        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker`
            walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
            updates inplace.
        state : :class:`pauxy.state.State`
            Simulation state.
        """

        mf = self.mf_shift
        ifac = self.iut_fac
        ufac = self.ut_fac
        nsq = self.mf_nsq
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nbasis)
        # Optimal field shift for real local energy approximation.
        shift = numpy.diag(walker.G[0])+numpy.diag(walker.G[1]) - mf
        xi_opt = -ifac*shift
        sxf = sum(xi-xi_opt)
        # Propagator for potential term with mean field and auxilary field shift.
        c_xf = cmath.exp(0.5*ufac*nsq-ifac*mf*sxf)
        EXP_VHS = numpy.exp(0.5*ufac*(1-2.0*mf)+ifac*(xi-xi_opt))
        nup = system.nup
        walker.phi[:,:nup] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,:nup])
        walker.phi[:,nup:] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,nup:])
        return c_xf

    def propagate_walker_free_continuous(self, walker, system, trial):
        r"""Free projection for continuous HS transformation.

        TODO: update if ever adapted to other model types.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        nup = system.nup
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.bt2)
        # Normally distributed random numbers.
        xfields =  numpy.random.normal(0.0, 1.0, system.nbasis)
        sxf = sum(xfields)
        # Constant, field dependent term emerging when subtracting mean-field.
        sc = 0.5*self.ut_fac*self.mf_nsq-self.iut_fac*self.mf_shift*sxf
        c_xf = cmath.exp(sc)
        # Potential propagator.
        s = self.iut_fac*xfields + 0.5*self.ut_fac*(1-2*self.mf_shift)
        bv = numpy.diag(numpy.exp(s))
        # 2. Apply potential projector.
        walker.phi[:,:nup] = bv.dot(walker.phi[:,:nup])
        walker.phi[:,nup:] = bv.dot(walker.phi[:,nup:])
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.bt2)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * c_xf

    def propagate_walker_constrained_continuous(self, walker, system, trial):
        r"""Wrapper function for propagation using continuous transformation.

        This applied the phaseless, local energy approximation and uses importance
        sampling.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """

        # 1. Apply kinetic projector.
        self.kinetic(walker.phi, system, self.bt2)
        # 2. Apply potential projector.
        cxf = self.two_body(walker, system, trial)
        # 3. Apply kinetic projector.
        self.kinetic(walker.phi, system, self.bt2)

        # Now apply phaseless, real local energy approximation
        walker.inverse_overlap(trial.psi)
        walker.greens_function(trial)
        E_L = walker.local_energy(system)[0].real
        # Check for large population fluctuations
        E_L = local_energy_bound(E_L, self.mean_local_energy,
                                 self.ebound)
        ot_new = walker.calc_otrial(trial.psi)
        # Walker's phase.
        dtheta = cmath.phase(cxf*ot_new/walker.ot)
        walker.weight = (walker.weight * math.exp(-0.5*self.dt*(walker.E_L+E_L))
                                       * max(0, math.cos(dtheta)))
        walker.E_L = E_L
        walker.ot = ot_new

class GenericContinuous:
    '''Base propagator class'''

    def __init__(self, options, qmc, system, trial):
        # Input options
        self.hs_type = 'continuous'
        self.free_projection = options.get('free_projection', False)
        self.exp_nmax = options.get('expansion_order', 6)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        # Mean field shifts (2,nchol_vec).
        self.mf_shift = 1j*numpy.einsum('lpq,spq->l', system.chol_vecs, trial.G)
        # Mean field shifted one-body propagator
        self.construct_one_body_propagator(qmc.dt, system.chol_vecs,
                                           system.h1e_mod)
        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore + 0.5*numpy.dot(self.mf_shift, self.mf_shift)
        self.mf_const_fac = cmath.exp(-self.dt*mf_core)
        # todo : ?
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz
        # Temporary array for matrix exponentiation.
        self.Temp = numpy.zeros(trial.psi[:,:system.nup].shape,
                                dtype=trial.psi.dtype)
        # Rotated cholesky vectors.
        # Assuming nup = ndown here
        rotated_up = numpy.einsum('rp,lpq->lrq',
                                  trial.psi[:,:system.nup].conj().T,
                                  system.chol_vecs)
        rotated_down = numpy.einsum('rp,lpq->lrq',
                                    trial.psi[:,system.nup:].conj().T,
                                    system.chol_vecs)
        self.rchol_vecs = numpy.array([rotated_up, rotated_down])
        # todo : remove
        self.chol_vecs = system.chol_vecs
        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_phaseless


    def construct_one_body_propagator(self, dt, chol_vecs, h1e_mod):
        shift = 1j*numpy.einsum('l,lpq->pq', self.mf_shift, chol_vecs)
        H1 = h1e_mod - numpy.array([shift,shift])
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_force_bias_opt(self, Gmod):
        vbias = 1j*numpy.einsum('slrp,spr->l', self.rchol_vecs, Gmod)
        return - self.sqrt_dt * (vbias-self.mf_shift)

    def construct_force_bias(self, G):
        vbias = numpy.einsum('lpq,pq->l', self.chol_vecs, G[0])
        vbias += numpy.einsum('lpq,pq->l', self.chol_vecs, G[1])
        return - self.sqrt_dt * (1j*vbias-self.mf_shift)

    def two_body(self, walker, system, trial):
        r"""Continuous Hubbard-Statonovich transformation for Hubbard model.

        Only requires M auxiliary fields.

        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker`
            walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
            updates inplace.
        state : :class:`pauxy.state.State`
            Simulation state.
        """
        # Construct walker modified Green's function.
        # walker.rotated_greens_function()
        walker.inverse_overlap(trial.psi)
        # walker.greens_function(trial)
        walker.rotated_greens_function()
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nchol_vec)
        # Optimal force bias.
        xbar = self.construct_force_bias_opt(walker.Gmod)
        # xbar2 = self.construct_force_bias(walker.G)
        shifted = xi - xbar
        # Constant factor arising from force bias and mean field shift
        c_xf = cmath.exp(-self.sqrt_dt*shifted.dot(self.mf_shift))
        # Constant factor arising from shifting the propability distribution.
        c_fb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))
        # Operator terms contributing to propagator.
        VHS = self.isqrt_dt*numpy.einsum('l,lpq->pq', shifted, system.chol_vecs)
        nup = system.nup
        # Apply propagator
        self.apply_exponential(walker.phi[:,:nup], VHS)
        self.apply_exponential(walker.phi[:,nup:], VHS)

        return (c_xf, c_fb, shifted)

    def apply_exponential(self, phi, VHS, debug=False):
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)
        numpy.copyto(self.Temp, phi)
        for n in range(1, self.exp_nmax+1):
            self.Temp = VHS.dot(self.Temp) / n
            phi += self.Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))

    def propagate_walker_free(self, walker, system, trial):
        r"""Free projection for continuous HS transformation.

        TODO: update if ever adapted to other model types.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        nup = system.nup
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.bt2)
        # Normally distributed random numbers.
        xfields =  numpy.random.normal(0.0, 1.0, system.nbasis)
        sxf = sum(xfields)
        # Constant, field dependent term emerging when subtracting mean-field.
        sc = 0.5*self.ut_fac*self.mf_nsq-self.iut_fac*self.mf_shift*sxf
        c_xf = cmath.exp(sc)
        # Potential propagator.
        s = self.iut_fac*xfields + 0.5*self.ut_fac*(1-2*self.mf_shift)
        bv = numpy.diag(numpy.exp(s))
        # 2. Apply potential projector.
        walker.phi[:,:nup] = bv.dot(walker.phi[:,:nup])
        walker.phi[:,nup:] = bv.dot(walker.phi[:,nup:])
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.bt2)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * c_xf

    def propagate_walker_phaseless(self, walker, system, trial):
        r"""Wrapper function for propagation using continuous transformation.

        This applied the phaseless, local energy approximation and uses importance
        sampling.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """

        # 1. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)
        # 2. Apply two_body propagator.
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # 3. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)

        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
        ot_new = walker.calc_otrial(trial.psi)
        # Walker's phase.
        importance_function = self.mf_const_fac*cxf*cfb*ot_new / walker.ot
        dtheta = cmath.phase(importance_function)
        cfac = max(0, math.cos(dtheta))
        rweight = abs(importance_function)
        walker.weight *= rweight * cfac
        walker.ot = ot_new
        walker.field_configs.push_full(xmxbar, cfac, importance_function/rweight)
