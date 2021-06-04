from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, \
    horizontal, region, interval, computation, stencil, \
    exp, log, sin, function, __INLINED
import gt4py as gt
import numpy as np

from nmwc_model import *
from nmwc_model.namelist import *

def make_1D_storage(nk):
    karray_shape = (1, 1, nk)
    return gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, karray_shape, dtype=DTYPE_FLOAT)

def make_2D_storage(ni, nj, dtype=DTYPE_FLOAT):
    array_shape = (ni, nj)
    return gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, array_shape, dtype=dtype)

def make_3D_storage(ni, nj, nk):
    array_shape = (ni, nj, nk)
    return gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, array_shape, dtype=DTYPE_FLOAT)

def to_K_array(k_storage):
    if isinstance(k_storage, gt.storage.Storage):
        dataview = k_storage[0, 0, :]
    else:
        dataview = k_storage
    nk = len(dataview)
    return gt.storage.from_array(dataview, BACKEND, DEFAULT_ORIGIN, shape=(nk,), mask=(False, False, True), dtype=DTYPE_FLOAT)

def view_storage(storage: gt.storage.Storage):
    storage.device_to_host()
    return storage.view(np.ndarray)

_x0 = (nxb-1)/2.0+1.0

def gaussian(x):
    return np.exp(-x*x)

def maketopo(topo: FIELD_FLOAT_IJ):
    topo_np = np.zeros(topo.shape[0])
    for ind_i in range(1, len(topo_np)-1):
        xl = (ind_i+1-1-_x0)*dx
        xm = (ind_i+1-_x0)*dx
        xr = (ind_i+1+1-_x0)*dx
        toponfl = topomx * gaussian(xl / topowd)
        toponfm = topomx * gaussian(xm / topowd)
        toponfr = topomx * gaussian(xr / topowd)
        topo_np[ind_i] = toponfm + 0.25*(toponfl-2*toponfm+toponfr)
    topo[:,0] = topo_np[:]

@stencil(backend=BACKEND, rebuild=REBUILD)
def makeprofile_1D(th0: FIELD_FLOAT,
                   exn0: FIELD_FLOAT,
                   mtg0: FIELD_FLOAT,
                   prs0: FIELD_FLOAT,
                   u0: FIELD_FLOAT):
    with computation(FORWARD):
        with interval(0, 1):
            th0 = th00
            exn0 = exn00
            mtg0 = g*z00 + th00*exn00 + dth*exn00/2.0
            prs0 = pref*(exn00/cp)**cpdr
            u0 = u00
        with interval(1, None):
            th0 = th0[0,0,-1] + dth
            th0_2 = (th0+th0[0,0,-1])*(th0+th0[0,0,-1])
            exn0 = exn0[0,0,-1] - 16*(g*g)*dth/(4*bv00*bv00*th0_2)
            mtg0 = mtg0[0,0,-1] + dth*exn0
            prs0 = pref*(exn0/cp)**cpdr
            u0 = u00

@stencil(backend=BACKEND, rebuild=REBUILD)
def makeprofile_3D(sold: FIELD_FLOAT,
                   snow: FIELD_FLOAT,
                   uold: FIELD_FLOAT,
                   unow: FIELD_FLOAT,
                   mtg: FIELD_FLOAT,
                   prs0: FIELD_FLOAT_K,
                   u0: FIELD_FLOAT_K,
                   mtg0: FIELD_FLOAT_K):
    with computation(FORWARD), interval(0, -1):
        sold = -1.0/g*(prs0[1]-prs0)/dth
        snow = sold
        uold = u0
        unow = u0
        mtg = mtg0

@stencil(backend=BACKEND, rebuild=REBUILD)
def settau(tau: FIELD_FLOAT,
           ind_k: FIELD_FLOAT):
    with computation(FORWARD):
        with interval(0, -1):
            tmp = 0.0
            if ind_k > nz - nab:
                tmp = sin(0.5*np.pi*(ind_k-(nz-nab-1))/nab)
                tau = diff + (diffabs - diff) * tmp * tmp
            else:
                tau = diff
        with interval(-1, None):
            tau = tau[0,0,-1]

def setbnd(src: FIELD_FLOAT,
           bnd1: FIELD_FLOAT_K,
           bnd2: FIELD_FLOAT_K,
           *,
           nxb):
    bnd1[:] = src[0,0,:]
    bnd2[:] = src[nxb-1,0,:]

@function
def eswat1_func(T):
    C1 = 7.90298
    C2 = 5.02808 / 2.302585092994046  # log(10) = 2.3...
    C3 = 1.3816e-7
    C4 = 11.344
    C5 = 8.1328e-3
    C6 = 3.49149
    RMIXV = 373.16 / T
    ES = (-C1 * (RMIXV - 1)
          + C2 * log(RMIXV)  # log10 -> log
          - C3 * (10 ** (C4 * (1 - 1 / RMIXV)) - 1)
          + C5 * (10 ** (-C6 * (RMIXV - 1)) - 1))
    eswat = 1013.246 * 10 ** ES
    return eswat

@stencil(backend=BACKEND, rebuild=REBUILD)
def prog_isendens(sold: FIELD_FLOAT,
                  snow: FIELD_FLOAT,
                  snew: FIELD_FLOAT,
                  unow: FIELD_FLOAT,
                  *,
                  dtdx: DTYPE_FLOAT):
    with computation(FORWARD), interval(...):
        snew = sold - dtdx*( snow[1,0,0]*0.5*(unow[1,0,0]+unow[2,0,0]) -
                             snow[-1,0,0]*0.5*(unow[-1,0,0]+unow))

@stencil(backend=BACKEND, rebuild=REBUILD)
def prog_velocity(uold: FIELD_FLOAT,
                  unow: FIELD_FLOAT,
                  unew: FIELD_FLOAT,
                  mtg: FIELD_FLOAT,
                  *,
                  dtdx: DTYPE_FLOAT):
    with computation(FORWARD), interval(...):
        unew = uold - unow*dtdx*(unow[1,0,0]-unow[-1,0,0]) - \
                2*dtdx*(mtg-mtg[-1,0,0])

nr = 8
rel = np.array([1, 0.99, 0.95, 0.8, 0.5, 0.2, 0.05, 0.01])
rel[:nb] = 1.0

def make_rel_field(ni, nj, nx, nb):
    rel_field = make_2D_storage(ni, nj)
    rel_mask = make_2D_storage(ni, nj, dtype=DTYPE_INT)
    n = 2*nb+nx
    rel_field[:nr, 0] = rel[:]
    rel_field[n-nr:n,0] = np.flip(rel)
    rel_mask[:nr, 0] = 1
    rel_mask[n-nr:n,0] = -1
    return rel_field, rel_mask

"""
@stencil(backend=BACKEND, rebuild=REBUILD)
def relax_boundary(phi: FIELD_FLOAT,
                   phi1: FIELD_FLOAT_K,
                   phi2: FIELD_FLOAT_K,
                   rel: FIELD_FLOAT_IJ):
    with computation(FORWARD), interval(...):
        with horizontal(region[:nr,0]):
            phi = phi1*rel + phi*(1-rel)
        with horizontal(region[nxb1-nr-1:nxb1,0]):
            phi = phi2*rel + phi*(1-rel)
"""

@stencil(backend=BACKEND, rebuild=REBUILD)
def relax_boundary_noregion(phi: FIELD_FLOAT,
                            phi1: FIELD_FLOAT_K,
                            phi2: FIELD_FLOAT_K,
                            rel: FIELD_FLOAT_IJ,
                            rel_mask: FIELD_INT_IJ):
    with computation(FORWARD), interval(...):
        if rel_mask > 0:
            phi = phi1*rel + phi*(1-rel)
        if rel_mask < 0:
            phi = phi2*rel + phi*(1-rel)

def relax_3D(phi: FIELD_FLOAT,
             phi1: FIELD_FLOAT_K,
             phi2: FIELD_FLOAT_K,
             *,
             nx: DTYPE_INT,
             nb: DTYPE_INT):
    nr = 8
    n = 2*nb+nx
    for i in range(nr):
        phi[i, 0, :] = phi1[:] * rel[i] + phi[i, 0, :] * (1 - rel[i])
        phi[n-1-i, 0, :] = phi2[:] * rel[i] + phi[n-1-i, 0, :] * (1 - rel[i])

def relax_2D(phi: FIELD_FLOAT_IJ,
             phi1: DTYPE_FLOAT,
             phi2: DTYPE_FLOAT,
             *,
             nx: DTYPE_INT,
             nb: DTYPE_INT):
    nr = 8
    n = 2*nb+nx
    for i in range(nr):
        phi[i, 0] = phi1 * rel[i] + phi[i, 0] * (1 - rel[i])
        phi[n-1-i, 0] = phi2 * rel[i] + phi[n-1-i, 0] * (1 - rel[i])

@stencil(backend=BACKEND, rebuild=REBUILD)
def diffusion(phinew: FIELD_FLOAT,
              phibuf: FIELD_FLOAT,
              tau: FIELD_FLOAT_K,
              *,
              diffvert: DTYPE_FLOAT):
    with computation(FORWARD), interval(0, -1):
        if tau > 0:
            phibuf = phinew + tau*(phinew[-1,0,0]-2*phinew+phinew[1,0,0])/4.0
    with computation(FORWARD), interval(0, -1):
        if tau > 0:
            phinew = phibuf

@stencil(backend=BACKEND, rebuild=REBUILD)
def diag_pressure(snew: FIELD_FLOAT,
                  prs: FIELD_FLOAT,
                  prs0: FIELD_FLOAT_K,
                  *,
                  dth: DTYPE_FLOAT):
    with computation(BACKWARD):
        with interval(-1, None):
            prs = prs0
        with interval(0, -1):
            prs = prs[0,0,1] + g*dth*snew

@stencil(backend=BACKEND, rebuild=REBUILD)
def diag_montgomery(exn: FIELD_FLOAT,
                    mtg: FIELD_FLOAT,
                    prs: FIELD_FLOAT,
                    topo: FIELD_FLOAT_IJ,
                    *,
                    topofact: DTYPE_FLOAT,
                    dth: DTYPE_FLOAT,
                    th00: DTYPE_FLOAT):
    with computation(FORWARD), interval(...):
        exn = cp*(prs/pref)**rdcp
    with computation(FORWARD):
        with interval(0, 1):
            mtg = g*topo*topofact + (th00+dth/2.0)*exn
        with interval(1, None):
            mtg = mtg[0,0,-1] + dth*exn

@stencil(backend=BACKEND, rebuild=REBUILD)
def diag_height(zhtnow: FIELD_FLOAT,
                exn: FIELD_FLOAT,
                prs: FIELD_FLOAT,
                th0: FIELD_FLOAT_K,
                topo: FIELD_FLOAT_IJ,
                *,
                topofact: DTYPE_FLOAT):
    with computation(FORWARD):
        with interval(0, 1):
            zhtnow = topofact*topo
        with interval(1, None):
            zhtnow = zhtnow[0,0,-1] - rdcp*(
                th0[-1]*exn[0,0,-1]+th0*exn
            )*(prs - prs[0,0,-1])/(g*(prs[0,0,-1]+prs))

def makeoutput(unow: FIELD_FLOAT,
               snow: FIELD_FLOAT,
               zhtnow: FIELD_FLOAT,
               its_out: DTYPE_INT,
               Z: np.ndarray,
               U: np.ndarray,
               S: np.ndarray):
    zhtnow.device_to_host()
    unow.device_to_host()
    snow.device_to_host()
    for j in range(nz):
        Z[its_out, j, :nx] = zhtnow[nb:nb+nx, 0, j]
        U[its_out, j, :nx] = 0.5 * unow[nb:nb+nx,0,j]
        U[its_out, j, :nx] += 0.5 * unow[nb+1:nb + nx + 1,0,j]
        S[its_out, j, :nx] = snow[nb:nb+nx, 0, j]
    Z[its_out, nz, :nx] = zhtnow[nb:nb+nx, 0, nz]
