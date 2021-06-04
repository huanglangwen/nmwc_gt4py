# -*- coding: utf-8 -*-
import numpy as np

from tests import conf
from nmwc_model import BACKEND, DEFAULT_ORIGIN
import gt4py as gt

def make_storage(np_arr: np.ndarray, nx, nb, nz):
    nxb1 = nx + 2*nb + 1
    nz1 = nz + 1
    len_x, len_y = np_arr.shape
    storage = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, (nxb1, 1, nz1), np_arr.dtype)
    storage[:len_x, :, :len_y] = np_arr[:, np.newaxis, :]
    return storage

def make_k_storage(np_arr: np.ndarray, nz):
    nz1 = nz + 1
    len_z = len(np_arr)
    storage = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, (nz1,), np_arr.dtype, mask=(False, False, True))
    storage[:len_z] = np_arr[:]
    return storage

def make_ij_storage(np_arr: np.ndarray, nx, nb):
    nxb1 = nx + 2*nb + 1
    len_x = len(np_arr)
    storage = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, (nxb1,1), np_arr.dtype)
    storage[:len_x, :] = np_arr[:]
    return storage

def get_random_int(min_value=0, max_value=100):
    return np.random.randint(min_value, max_value, dtype=int)


def get_random_nx(min_value=conf.nx_min, max_value=conf.nx_max):
    return get_random_int(min_value, max_value)


def get_random_nb(min_value=conf.nb_min, max_value=conf.nb_max):
    return get_random_int(min_value, max_value)


def get_random_nz(min_value=conf.nz_min, max_value=conf.nz_max):
    return get_random_int(min_value, max_value)


def get_random_float(min_value=conf.field_min, max_value=conf.field_max):
    return min_value + (max_value - min_value) * np.random.rand(1).item()


def get_random_positive_float(max_value=conf.field_max):
    out = get_random_float(min_value=-max_value, max_value=max_value)
    return out if out > 0.0 else -out


def get_random_array_1d(n, min_value=conf.field_min, max_value=conf.field_max, sort=False):
    out = min_value + (max_value - min_value) * np.random.rand(n)
    return sorted(out) if sort else out


def get_random_array_2d(ni, nk, min_value=conf.field_min, max_value=conf.field_max):
    return min_value + (max_value - min_value) * np.random.rand(ni, nk)


def compare_arrays(a, b):
    assert np.allclose(a, b, equal_nan=True)
