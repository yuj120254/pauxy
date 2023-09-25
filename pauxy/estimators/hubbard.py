import numpy
def local_energy_hubbard_holstein_momentum(system, G, P, Lap, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard-Hostein model.

    Parameters
    ----------
    system : :class:`HubbardHolstein`
        System information for the HubbardHolstein model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    # T = kinetic_lang_firsov(system.t, system.gamma_lf, P, system.nx, system.ny, system.ktwist)

    Dp = numpy.array([numpy.exp(1j*system.gamma_lf*P[i]) for i in range(system.nbasis)])
    T = numpy.zeros_like(system.T, dtype=numpy.complex128)
    T[0] = numpy.diag(Dp).dot(system.T[0]).dot(numpy.diag(Dp.T.conj()))
    T[1] = numpy.diag(Dp).dot(system.T[1]).dot(numpy.diag(Dp.T.conj()))

    ke = numpy.sum(T[0] * G[0] + T[1] * G[1])

    sqrttwomw = numpy.sqrt(2.0 * system.m * system.w0)
    assert (system.gamma_lf * system.w0  == system.g * sqrttwomw)

    Ueff = system.U + system.gamma_lf**2 * system.w0 - 2.0 * system.g * system.gamma_lf * sqrttwomw

    if system.symmetric:
        pe = -0.5*Ueff*(G[0].trace() + G[1].trace())

    pe = Ueff * numpy.dot(G[0].diagonal(), G[1].diagonal())

    pe_ph = - 0.5 * system.w0 ** 2 * system.m * numpy.sum(Lap)
    ke_ph = 0.5 * numpy.sum(P*P) / system.m - 0.5 * system.w0 * system.nbasis
    
    rho = G[0].diagonal() + G[1].diagonal()
    
    e_eph = (system.gamma_lf**2 * system.w0 / 2.0 - system.g * system.gamma_lf * sqrttwomw) * numpy.sum(rho)

    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke+pe, ke_ph+pe_ph+e_eph)

def local_energy_hubbard_holstein(system, G, X, Lap, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard-Hostein model.

    Parameters
    ----------
    system : :class:`HubbardHolstein`
        System information for the HubbardHolstein model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"
    X : :class:`numpy.ndarray`
        Walker's phonon coordinate

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])

    if system.symmetric:
        pe = -0.5*system.U*(G[0].trace() + G[1].trace())

    pe = system.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    
    pe_ph = 0.5 * system.w0 ** 2 * system.m * numpy.sum(X * X)

    ke_ph = -0.5 * numpy.sum(Lap) / system.m - 0.5 * system.w0 * system.nbasis
    
    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = - system.g * numpy.sqrt(system.m * system.w0 * 2.0) * numpy.dot(rho, X)


    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke+pe, ke_ph+pe_ph+e_eph)


def local_energy_hubbard(system, G, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])
    # Todo: Stupid
    if system.symmetric:
        pe = -0.5*system.U*(G[0].trace() + G[1].trace())
    pe = system.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    return (ke + pe, ke, pe)


def local_energy_hubbard_ghf(system, Gi, weights, denom):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    denom : float
        Overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:system.nbasis,:system.nbasis], axis1=1, axis2=2)
    gdd = numpy.diagonal(Gi[:,system.nbasis:,system.nbasis:], axis1=1, axis2=2)
    gud = numpy.diagonal(Gi[:,system.nbasis:,:system.nbasis], axis1=1, axis2=2)
    gdu = numpy.diagonal(Gi[:,:system.nbasis,system.nbasis:], axis1=1, axis2=2)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('j,jk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)

def local_energy_hubbard_ghf_full(system, GAB, weights):
    r"""Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions for different SDs A and B.
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L, T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('ij,ijkl,kl->', weights, GAB, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(GAB[:,:,:system.nbasis,:system.nbasis], axis1=2,
                         axis2=3)
    gdd = numpy.diagonal(GAB[:,:,system.nbasis:,system.nbasis:], axis1=2,
                         axis2=3)
    gud = numpy.diagonal(GAB[:,:,system.nbasis:,:system.nbasis], axis1=2,
                         axis2=3)
    gdu = numpy.diagonal(GAB[:,:,:system.nbasis,system.nbasis:], axis1=2,
                         axis2=3)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('ij,ijk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)


def local_energy_multi_det(system, Gi, weights):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:,:system.nup], axis1=1,
                         axis2=2)
    gdd = numpy.diagonal(Gi[:,:,system.nup:], axis1=1,
                         axis2=2)
    pe = system.U * numpy.einsum('j,jk->', weights, guu*gdd) / denom
    return (ke+pe, ke, pe)


def fock_hubbard(system, P):
    """Hubbard Fock Matrix
        F_{ij} = T_{ij} + U(<niu>nid + <nid>niu)_{ij}
    """
    niu = numpy.diag(P[0].diagonal())
    nid = numpy.diag(P[1].diagonal())
    return system.T + system.U*numpy.array([nid,niu])

def order_parameter_hubbard_holstein(system, Psi):
    r"""Calculate order parameters for the Hubbard-Hostein model.

    Parameters
    ----------
    system : :class:`HubbardHolstein`
        System information for the HubbardHolstein model.
    Psi : :class:`numpy.ndarray`
        Walker's wavefunction

    Returns
    -------
    (SDW_OP, CDW_OP): tuple
        SDW and CDW order parameters
    """
    SDW_OP = 0
    CDW_OP = 0

    psi_up = Psi[:,:8]
    psi_down = Psi[:,8:]
    density_up = numpy.diag(psi_up.dot((psi_up.conj()).T))
    density_down = numpy.diag(psi_down.dot((psi_down.conj()).T))

    #print("Up densities:", density_up)
    #print("Down densities:", density_down)
    #print("Sum of Den up", numpy.sum(numpy.sum(density_up)))
    #print("Sum of Den down", numpy.sum(numpy.sum(density_up)))

    niup_final = numpy.reshape(density_up, (4, 4))
    nidown_final = numpy.reshape(density_down, (4, 4))
    nitotal_final = niup_final + nidown_final

    for i in range(4):
        for j in range(4):
            SDW_OP += numpy.abs(niup_final[i,j] - nidown_final[i,j])/16

    for i in range(4):
        for j in range(4):
            CDW_OP += numpy.abs(nitotal_final[i,j] - nitotal_final[(i+1)%4,j])/32
            CDW_OP += numpy.abs(nitotal_final[i,j] - nitotal_final[i, (j+1)%4])/32


    #print("SDW order parameter:", SDW_OP)
    #print("CDW order parameter:", CDW_OP)
 
    return (SDW_OP, CDW_OP)

