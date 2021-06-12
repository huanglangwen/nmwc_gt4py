# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import pytest

from nmwc_model.kernels import diffusion

from tests import utils
from tests.utils import make_storage, make_k_storage, make_ij_storage

def test_dry():
    # load reference data
    ds = np.load("baseline_datasets/test_diffusion/test_dry.npz")

    # hack
    nx = int(ds["nx"])
    nx1 = nx+1
    nb = int(ds["nb"])
    nz = ds["unew"].shape[1]
    tau = make_k_storage(ds["tau"], nz)
    unew = make_storage(ds["unew"], nx, nb, nz)
    snew = make_storage(ds["snew"], nx, nb, nz)
    ubuf = make_storage(np.zeros_like(ds["unew"]), nx, nb, nz)
    sbuf = make_storage(np.zeros_like(ds["snew"]), nx, nb, nz)

    # run user code
    diffusion(snew, sbuf, tau, origin=(nb, 0, 0), domain=(nx, 1, nz))
    diffusion(unew, ubuf, tau, origin=(nb, 0, 0), domain=(nx1, 1, nz))
    snew.device_to_host()
    unew.device_to_host()

    # validation
    # array[1, 0, :]
    utils.compare_arrays(unew[nb:nx1+nb,0,:nz].view(np.ndarray), ds["unew_val"][nb:nx1+nb,:])
    utils.compare_arrays(snew[nb:nx+nb,0,:nz].view(np.ndarray), ds["snew_val"][nb:nx+nb,:])


def test_dry_periodic():
    # load reference data
    ds = np.load("baseline_datasets/test_diffusion/test_dry_periodic.npz")

    # hack
    diffusion.__dict__["nx"] = ds["nx"]
    diffusion.__dict__["nb"] = ds["nb"]
    diffusion.__dict__["imoist"] = 0
    diffusion.__dict__["irelax"] = 0

    # run user code
    unew, snew = diffusion.horizontal_diffusion(ds["tau"], ds["unew"], ds["snew"])

    # validation
    utils.compare_arrays(unew, ds["unew_val"])
    utils.compare_arrays(snew, ds["snew_val"])


def test_moist():
    # load reference data
    ds = np.load("baseline_datasets/test_diffusion/test_moist.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    tau = ds["tau"]
    unew, unew_val = ds["unew"], ds["unew_val"]
    snew, snew_val = ds["snew"], ds["snew_val"]
    qvnew, qvnew_val = ds["qvnew"], ds["qvnew_val"]
    qcnew, qcnew_val = ds["qcnew"], ds["qcnew_val"]
    qrnew, qrnew_val = ds["qrnew"], ds["qrnew_val"]
    ncnew, ncnew_val = ds["ncnew"], ds["ncnew_val"]
    nrnew, nrnew_val = ds["nrnew"], ds["nrnew_val"]

    # prepare input data
    unew_dc = deepcopy(unew)
    snew_dc = deepcopy(snew)
    qvnew_dc = deepcopy(qvnew)
    qcnew_dc = deepcopy(qcnew)
    qrnew_dc = deepcopy(qrnew)

    #
    # imicrophys = 0,1
    #
    # hack
    diffusion.__dict__["nx"] = nx
    diffusion.__dict__["nb"] = nb
    diffusion.__dict__["imoist"] = 1
    diffusion.__dict__["imoist_diff"] = 1
    diffusion.__dict__["irelax"] = 1
    diffusion.__dict__["imicrophys"] = 0

    # run user code
    out_list = diffusion.horizontal_diffusion(tau, unew, snew, qvnew, qcnew, qrnew)

    # validation
    assert len(out_list) == 5
    utils.compare_arrays(out_list[0], unew_val)
    utils.compare_arrays(out_list[1], snew_val)
    utils.compare_arrays(out_list[2], qvnew_val)
    utils.compare_arrays(out_list[3], qcnew_val)
    utils.compare_arrays(out_list[4], qrnew_val)

    #
    # imicrophys = 2
    #
    # hack
    diffusion.__dict__["imicrophys"] = 2

    # run user code
    out_list = diffusion.horizontal_diffusion(
        tau, unew_dc, snew_dc, qvnew_dc, qcnew_dc, qrnew_dc, ncnew, nrnew
    )

    # validation
    assert len(out_list) == 7
    utils.compare_arrays(out_list[0], unew_val)
    utils.compare_arrays(out_list[1], snew_val)
    utils.compare_arrays(out_list[2], qvnew_val)
    utils.compare_arrays(out_list[3], qcnew_val)
    utils.compare_arrays(out_list[4], qrnew_val)
    utils.compare_arrays(out_list[5], ncnew_val)
    utils.compare_arrays(out_list[6], nrnew_val)


def test_moist_periodic():
    # load reference data
    ds = np.load("baseline_datasets/test_diffusion/test_moist_periodic.npz")
    nx = ds["nx"]
    nb = ds["nb"]
    tau = ds["tau"]
    unew, unew_val = ds["unew"], ds["unew_val"]
    snew, snew_val = ds["snew"], ds["snew_val"]
    qvnew, qvnew_val = ds["qvnew"], ds["qvnew_val"]
    qcnew, qcnew_val = ds["qcnew"], ds["qcnew_val"]
    qrnew, qrnew_val = ds["qrnew"], ds["qrnew_val"]
    ncnew, ncnew_val = ds["ncnew"], ds["ncnew_val"]
    nrnew, nrnew_val = ds["nrnew"], ds["nrnew_val"]

    # prepare input data
    unew_dc = deepcopy(unew)
    snew_dc = deepcopy(snew)
    qvnew_dc = deepcopy(qvnew)
    qcnew_dc = deepcopy(qcnew)
    qrnew_dc = deepcopy(qrnew)

    #
    # imicrophys = 0,1
    #
    # hack
    diffusion.__dict__["nx"] = nx
    diffusion.__dict__["nb"] = nb
    diffusion.__dict__["imoist"] = 1
    diffusion.__dict__["imoist_diff"] = 1
    diffusion.__dict__["irelax"] = 0
    diffusion.__dict__["imicrophys"] = 0

    # run user code
    out_list = diffusion.horizontal_diffusion(tau, unew, snew, qvnew, qcnew, qrnew)

    # validation
    assert len(out_list) == 5
    utils.compare_arrays(out_list[0], unew_val)
    utils.compare_arrays(out_list[1], snew_val)
    utils.compare_arrays(out_list[2], qvnew_val)
    utils.compare_arrays(out_list[3], qcnew_val)
    utils.compare_arrays(out_list[4], qrnew_val)

    #
    # imicrophys = 2
    #
    # hack
    diffusion.__dict__["imicrophys"] = 2

    # run user code
    out_list = diffusion.horizontal_diffusion(
        tau, unew_dc, snew_dc, qvnew_dc, qcnew_dc, qrnew_dc, ncnew, nrnew
    )

    # validation
    assert len(out_list) == 7
    utils.compare_arrays(out_list[0], unew_val)
    utils.compare_arrays(out_list[1], snew_val)
    utils.compare_arrays(out_list[2], qvnew_val)
    utils.compare_arrays(out_list[3], qcnew_val)
    utils.compare_arrays(out_list[4], qrnew_val)
    utils.compare_arrays(out_list[5], ncnew_val)
    utils.compare_arrays(out_list[6], nrnew_val)


if __name__ == "__main__":
    pytest.main([__file__])
