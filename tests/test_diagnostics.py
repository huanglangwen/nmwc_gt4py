# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model.kernels import diag_pressure, diag_height, diag_montgomery

from tests import utils
from tests.utils import make_storage, make_k_storage, make_ij_storage

def test_pressure():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_pressure.npz")

    # prepare input data
    nx = ds["snew"].shape[0]
    nb = 0
    nz = int(ds["nz"])
    dth = float(ds["dth"])
    nxb1 = nx+2*nb+1
    nz1 = nz+1
    prs = make_storage(np.zeros_like(ds["prs_val"]), nx, nb, nz)
    snew = make_storage(ds["snew"], nx, nb, nz)
    prs0 = make_k_storage(ds["prs0"], nz)

    diag_pressure(snew, prs, prs0, dth=dth, origin=(0, 0, 0), domain=(nxb1, 1, nz1))
    prs.device_to_host()
    # validation
    utils.compare_arrays(prs[nb:nx+nb, 0, :nz1].view(np.ndarray), ds["prs_val"][nb:nx+nb,:])


def test_montgomery():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_montgomery.npz")

    # prepare input data
    nx = ds["mtg_val"].shape[0]
    nb = 0
    nz = int(ds["nz"])
    dth = float(ds["dth"])
    nxb1 = nx+2*nb+1
    nz1 = nz+1
    prs = make_storage(ds["prs"], nx, nb, nz)
    mtg = make_storage(np.zeros_like(ds["mtg_val"]), nx, nb, nz)
    exn = make_storage(np.zeros_like(ds["mtg_val"]), nx, nb, nz)
    topo = make_ij_storage(ds["topo"], nx, nb)
    topofact = float(ds["topofact"])

    # run user code
    diag_montgomery(exn, mtg, prs, topo, topofact=topofact, dth=dth, th00=ds["th0"][0], origin=(0, 0, 0), domain=(nxb1, 1, nz1))
    exn.device_to_host()
    mtg.device_to_host()
    # validation
    utils.compare_arrays(exn[nb:nx+nb,0,:nz1].view(np.ndarray), ds["exn_val"][nb:nx+nb,:])
    utils.compare_arrays(mtg[nb:nx+nb,0,:nz].view(np.ndarray), ds["mtg_val"][nb:nx+nb,:])


def test_height():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_height.npz")

    # prepare input data
    nx = ds["prs"].shape[0]
    nb = 0
    nz = int(ds["nz"])
    nxb1 = nx+2*nb+1
    nz1 = nz+1
    prs = make_storage(ds["prs"], nx, nb, nz)
    exn = make_storage(ds["exn"], nx, nb, nz)
    zht = make_storage(np.zeros_like(ds["zht_val"]), nx, nb, nz)
    th0 = make_k_storage(ds["th0"], nz)
    topo = make_ij_storage(ds["topo"], nx, nb)
    topofact = float(ds["topofact"])

    # run user code
    diag_height(zht, exn, prs, th0, topo, topofact=topofact, origin=(0, 0, 0), domain=(nxb1, 1, nz1))
    zht.device_to_host()
    # validation
    utils.compare_arrays(zht[nb:nx+nb,0,:], ds["zht_val"][nb:nx+nb,:])

"""
def test_density_and_temperature():
    # load reference data
    ds = np.load("baseline_datasets/test_diagnostics/test_density_and_temperature.npz")

    # hack
    diagnostics.__dict__["nz"] = ds["nz"]

    # run user code
    rho, temp = diagnostics.diag_density_and_temperature(
        ds["s"], ds["exn"], ds["zht"], ds["th0"]
    )

    # validation
    utils.compare_arrays(rho, ds["rho_val"])
    utils.compare_arrays(temp, ds["temp_val"])
"""

if __name__ == "__main__":
    pytest.main([__file__])
