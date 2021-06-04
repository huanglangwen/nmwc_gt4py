# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nmwc_model.kernels import prog_isendens, prog_velocity


from tests import utils
from tests.utils import make_storage


def test_prog_isendens():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_isendens.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    nb = int(ds["nb"])
    nx = int(ds["nx"])
    nz = int(ds["nz"])
    sold = make_storage(ds["sold"], nx, nb, nz)
    snow = make_storage(ds["snow"], nx, nb, nz)
    snew = make_storage(np.zeros_like(ds["snow"]), nx, nb, nz)
    unow = make_storage(ds["unow"], nx, nb, nz)

    prog_isendens(sold, snow, snew, unow, dtdx = dtdx, origin = (nb, 0, 0), domain = (nx, 1, nz))
    snew.device_to_host()

    # validation
    utils.compare_arrays(snew[nb:nx+nb, 0,:nz].view(np.ndarray), ds["snew_val"][nb:nx+nb, :])

def test_prog_velocity():
    # load reference data
    ds = np.load("baseline_datasets/test_prognostics/test_prog_velocity.npz")

    # prepare input data
    dtdx = ds["dt"] / ds["dx"]

    nb = int(ds["nb"])
    nx = int(ds["nx"])
    nx1 = nx+1
    nz = int(ds["nz"])
    uold = make_storage(ds["uold"], nx, nb, nz)
    unow = make_storage(ds["unow"], nx, nb, nz)
    unew = make_storage(np.zeros_like(ds["unow"]), nx, nb, nz)
    mtg = make_storage(ds["mtg"], nx, nb, nz)

    # run user code
    prog_velocity(uold, unow, unew, mtg, dtdx=dtdx, origin=(nb, 0, 0), domain=(nx1, 1, nz))
    unew.device_to_host()

    # validation
    utils.compare_arrays(unew[nb:nx1+nb, 0, :nz].view(np.ndarray), ds["unew_val"][nb:nx1+nb,:])


if __name__ == "__main__":
    pytest.main([__file__])
