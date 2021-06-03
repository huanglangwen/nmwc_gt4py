import numpy as np

from nmwc_model.kernels import *
from nmwc_model.namelist import *
from nmwc_model import BACKEND, DEFAULT_ORIGIN, DTYPE_FLOAT
from nmwc_model.output import write_output

import gt4py as gt



th0 = make_1D_storage(nz1)
exn0 = make_1D_storage(nz1)
mtg0 = make_1D_storage(nz1)
prs0 = make_1D_storage(nz1)
u0 = make_1D_storage(nz1)
tau = make_1D_storage(nz1)
ind_k = make_1D_storage(nz1)
ind_k[0,0,:] = np.arange(nz1)

topo = make_2D_storage(nxb1,1)
ind_i = make_2D_storage(nxb1,1)
ind_i[:,0] = np.arange(nxb1)

sold = make_3D_storage(nxb1, 1, nz1)
snow = make_3D_storage(nxb1, 1, nz1)
snew = make_3D_storage(nxb1, 1, nz1)
uold = make_3D_storage(nxb1, 1, nz1)
unow = make_3D_storage(nxb1, 1, nz1)
unew = make_3D_storage(nxb1, 1, nz1)

mtg = make_3D_storage(nxb1, 1, nz1)
prs = make_3D_storage(nxb1, 1, nz1)
exn = make_3D_storage(nxb1, 1, nz1)
zhtnow = make_3D_storage(nxb1, 1, nz1)
rel_field, rel_mask = make_rel_field(nxb1, nz1, nx, nb)
rel_field1, rel_mask1 = make_rel_field(nxb1, nz1, nx1, nb)

T = np.arange(1, nout + 1)
Z = np.zeros((nout, nz1, nx))
U = np.zeros((nout, nz, nx))
S = np.zeros((nout, nz, nx))

makeprofile_1D(th0, exn0, mtg0, prs0, u0)
settau(tau, ind_k)
th0 = to_K_array(th0)
exn0 = to_K_array(exn0)
mtg0 = to_K_array(mtg0)
prs0 = to_K_array(prs0)
u0 = to_K_array(u0)
tau = to_K_array(tau)
makeprofile_3D(sold, snow, uold, unow, mtg, prs0, u0, mtg0)


maketopo(topo)
if irelax == 1:
    relax_2D(topo, topo[0,0], topo[nxb-1,0], nx=nx, nb=nb)
    sbnd1 = to_K_array(np.zeros(nz1))
    sbnd2 = to_K_array(np.zeros(nz1))
    ubnd1 = to_K_array(np.zeros(nz1))
    ubnd2 = to_K_array(np.zeros(nz1))
    setbnd(snow, sbnd1, sbnd2, nxb=nxb)
    setbnd(unow, ubnd1, ubnd2, nxb=nxb1)

its_out = 0
for its in range(1, int(nts+1)):
    time = its*dt
    topofact = np.float64(min(1.0, float(time) / topotim))
    if its == 1:
        dtdx = dt/dx/2.0
    else:
        dtdx = dt/dx

    prog_isendens(sold, snow, snew, unow, dtdx=dtdx, origin=(nb, 0, 0), domain=(nx, 1, nz1))
    prog_velocity(uold, unow, unew, mtg, dtdx=dtdx, origin=(nb, 0, 0), domain=(nx1, 1, nz1))
    if irelax == 1:
        relax_boundary(snew, sbnd1, sbnd2, rel_field)
        relax_boundary(unew, ubnd1, ubnd2, rel_field1)
        #relax_3D(snew, sbnd1, sbnd2, nx=nx, nb=nb)
        #relax_3D(unew, ubnd1, ubnd2, nx=nx1, nb=nb)

    diffusion(snew, sold, tau, diffvert=diffvert, origin=(nb, 0, 0), domain=(nx, 1, nz1))
    diffusion(unew, uold, tau, diffvert=diffvert, origin=(nb, 0, 0), domain=(nx1, 1, nz1))

    diag_pressure(snew, prs, prs0, origin=(0, 0, 0), domain=(nxb1, 1, nz1))
    diag_montgomery(exn, mtg, prs, topo, topofact=topofact, origin=(0, 0, 0), domain=(nxb1, 1, nz1))
    diag_height(zhtnow, exn, prs, th0, topo, topofact=topofact, origin=(0, 0, 0), domain=(nxb1, 1, nz1))

    uold, unow, unew = unow, unew, uold
    sold, snow, snew = snow, snew, sold

    if iprtcfl == 1:
        u_max = np.amax(view_storage(unow[nb:nb+nx1,0,:nz]))
        cfl_max = u_max * dtdx
        print("============================================================\n")
        print(f"T: {time} CFL MAX: {cfl_max} U MAX: {u_max} m/s \n")
        if cfl_max > 1:
            print("!!! WARNING: CFL larger than 1 !!!\n")
        elif np.isnan(cfl_max):
            print("!!! MODEL ABORT: NaN values !!!\n")
        print("============================================================\n")

    if np.mod(its, iout) == 0:
        T[its_out] = time
        makeoutput(unow, snow, zhtnow, its_out, Z, U, S)
        its_out += 1

write_output(nout, Z, U, S, T)