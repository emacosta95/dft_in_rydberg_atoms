# import the sparse eigensolver
import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quspin
import torch
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.operators import quantum_LinearOperator, quantum_operator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm import trange
from quspin.tools.lanczos import lanczos_full, lanczos_iter, lin_comb_Q_T, expm_lanczos


def lanczos_method(hamiltonian: quspin.operators.hamiltonian, basis: quspin.basis):
    """Quspin Code for the Lanczos method --> http://weinbe58.github.io/QuSpin/examples/example20.html#example20-label"""
    # apply Lanczos
    # initial state for Lanczos algorithm
    v0 = np.random.normal(0, 1, size=basis.Ns)
    v0 = v0 / np.linalg.norm(v0)
    #
    m_GS = 10  # Krylov subspace dimension
    #
    # Lanczos finds the largest-magnitude eigenvalues:
    e, v, q_t = lanczos_full(hamiltonian, v0, m_GS)
    #
    #
    # compute ground state vector
    psi_GS_lanczos = lin_comb_Q_T(v[:, 0], q_t)

    return e, psi_GS_lanczos


def density_of_functional_pbc(
    psi: np.array, l: int, basis: quspin.basis, j_1: float, j_2: float, check_2nn: bool
):
    m = {}
    exp_m = []
    for i in range(l):

        if check_2nn:
            coupling = [[j_1, i, (i + 1) % l], [j_2, i, (i + 2) % l]]
        else:
            coupling = [[j_1, i, (i + 1) % l]]
        op = ["zz", coupling]
        static = [op]
        dynamic = []
        m = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        exp_m.append(m.expt_value(psi))
    return exp_m


def compute_magnetization(psi: np.array, l: int, basis: quspin.basis, direction: str):
    # define the connection
    m = {}
    exp_m = []
    for i in range(l):
        coupling = [[1, i]]
        op = [direction, coupling]
        static = [op]
        dynamic = []
        m = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.float64,
            check_symm=True,
            check_herm=False,
            check_pcon=False,
        )
        exp_m.append(m.expt_value(psi))
    return exp_m


def compute_correlation(psi: np.array, l: int, basis: quspin.basis, direction: str):
    for i in range(l):
        exp_m_i = []
        for j in range(l):
            coupling = [[1, i, j]]
            op = [direction, coupling]
            static = [op]
            dynamic = []
            m = quantum_LinearOperator(
                static,
                basis=basis,
                dtype=np.float64,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
            )
            exp_m_i.append(m.expt_value(psi))
        exp_m_i = np.asarray(exp_m_i)

        if i == 0:
            exp_m = exp_m_i.reshape(1, -1)
        else:
            exp_m = np.append(exp_m, exp_m_i.reshape(1, -1), axis=0)

    return exp_m


def transverse_ising_sparse_DFT1d(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    omega: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        omega_h = [[omega, k] for k in range(l)]
        if check_2nn:
            static = [["zz", j_1nn], ["zz", j_2nn], [
                "z", h], ["x", omega_h]]  # , ["x", eps_h]]
        else:
            static = [["zz", j_1nn], ["z", h], ["x", omega_h]]
        dynamic = []
        ham = hamiltonian(
            static,
            dynamic,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

        z = np.asarray(z)
        f_dens = density_of_functional_pbc(
            psi_0, l=l, basis=basis, j_1=j1, j_2=j2, check_2nn=check_2nn
        )
        f_dens = np.asarray(f_dens)
        if r == 0:
            zs = z.reshape(1, -1)
            fs_dens = f_dens.reshape(1, -1)
            es = e
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            fs_dens = np.append(fs_dens, f_dens.reshape(1, -1), axis=0)
            es = np.append(es, e)
    if z_2:
        text_z2 = "_augmentation"
        fs_dens = np.append(fs_dens, fs_dens, axis=0)
        zs = np.append(zs, -1 * zs, axis=0)

    dir = "data/dataset_1nn/"
    if check_2nn:
        dir = "data/dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" +
        text_field + f"_{fs_dens.shape[0]}_n"
    )

    return file_name, hs, zs, fs_dens, es


# def transverse_ising_sparse_DFT_lanczos_method(
#     h_max: int,
#     hs: np.ndarray,
#     n_dataset: int,
#     l: int,
#     j1: float,
#     j2: float,
#     pbc: bool,
#     z_2: bool,
#     file_name: str,
#     check_2nn: bool,
#     eps_breaking: float,
# ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

#     # file_name information
#     text_z2 = ""
#     text_field = f"{h_max:.2f}_h"

#     hs = hs
#     # the basis of the representation
#     basis = spin_basis_1d(l)

#     # the coupling terms
#     if pbc:
#         j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
#         if check_2nn:
#             j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
#     else:
#         j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
#         if check_2nn:
#             j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

#     for r in trange(n_dataset):

#         h = [[hs[r, k], k] for k in range(l)]  # external field
#         eps_h = [[eps_breaking, k] for k in range(l)]
#         if check_2nn:
#             # , ["x", eps_h]]
#             static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]
#         else:
#             static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
#         dynamic = []
#         ham = hamiltonian(
#             static,
#             dynamic,
#             basis=basis,
#             dtype=np.float64,
#             check_symm=False,
#             check_herm=False,
#             check_pcon=False,
#         )
#         # e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)
#         e, psi_0 = lanczos_method(hamiltonian=ham, basis=basis)
#         z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

#         z = np.asarray(z)
#         f_dens = density_of_functional_pbc(
#             psi_0, l=l, basis=basis, j_1=j1, j_2=j2, check_2nn=check_2nn
#         )
#         f_dens = np.asarray(f_dens)
#         if r == 0:
#             zs = z.reshape(1, -1)
#             fs_dens = f_dens.reshape(1, -1)
#             es = e
#         else:
#             zs = np.append(zs, z.reshape(1, -1), axis=0)
#             fs_dens = np.append(fs_dens, f_dens.reshape(1, -1), axis=0)
#             es = np.append(es, e)
#     if z_2:
#         text_z2 = "_augmentation"
#         fs_dens = np.append(fs_dens, fs_dens, axis=0)
#         zs = np.append(zs, -1 * zs, axis=0)

#     dir = "data/dataset_1nn/"
#     if check_2nn:
#         dir = "data/dataset_2nn/"

#     file_name = (
#         dir + file_name + text_z2 + f"_{l}_l_" +
#         text_field + f"_{fs_dens.shape[0]}_n"
#     )

#     return file_name, hs, zs, fs_dens, es


def transverse_ising_sparse_Den2Magn_dataset(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    omega: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        omega_h = [[omega, k] for k in range(l)]
        if check_2nn:
            static = [["zz", j_1nn], ["zz", j_2nn], [
                "z", h], ["x", omega_h]]  # , ["x", eps_h]]
        else:
            static = [["zz", j_1nn], ["z", h], ["x", omega_h]]
        dynamic = []
        ham = hamiltonian(
            static,
            dynamic,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        x = compute_magnetization(psi_0, l=l, basis=basis, direction="x")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

        z = np.asarray(z)
        x = np.asarray(x)
        if r == 0:
            zs = z.reshape(1, -1)
            xs = x.reshape(1, -1)
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            xs = np.append(xs, x.reshape(1, -1), axis=0)

    dir = "data/den2magn_dataset_1nn/"
    if check_2nn:
        dir = "data/den2magn_dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" +
        text_field + f"_{xs.shape[0]}_n"
    )

    return file_name, zs, xs


def transverse_ising_sparse_h_k_mapping_check(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    pbc: bool,
    omega: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    # j_nn = []
    # for s in range(1, 5):
    #     print('coupling=', (j1/s)**6)
    #     if pbc:
    #         j_nn.extend([[(j1/s)**6, i, (i + s) % l]
    #                     for i in range(l)])  # pbc
    #     else:
    #         j_nn.extend([[(j1/s)**6, i, (i + s) % l]
    #                     for i in range(l)])  # pbc
    if pbc:

        j_nn = [[(j1), i, (i+1) % l] for i in range(l)]
        j_nn.extend([[(j1/2)**6, i, (i+2) % l] for i in range(l)])
        j_nn.extend([[(j1/3)**6, i, (i+3) % l] for i in range(l)])
        j_nn.extend([[(j1/4)**6, i, (i+4) % l] for i in range(l)])

    else:
        j_nn = [[1, i, (i+1)] for i in range(l-1)]
        j_nn.extend([[(1/2)**6, i, (i+2)] for i in range(l-2)])
        j_nn.extend([[(1/3)**6, i, (i+3)] for i in range(l-3)])
        j_nn.extend([[(1/4)**6, i, (i+4)] for i in range(l-4)])

    for r in trange(n_dataset):
        print(hs[r])

        h = [[hs[r, k], k] for k in range(l)]  # external field
        omega_h = [[omega, k] for k in range(l)]
        static = [["xx", j_nn], [
            "x", h], ["z", omega_h]]
        dynamic = []
        ham = hamiltonian(
            static,
            dynamic,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        # print(ham.as_dense_format())
        e_min, psi_0 = ham.eigsh(k=1, which="SA", maxiter=1E4,)
        # psi_0 = psi_0.reshape((-1,))
        # e_scipy, psi_scipy = eigsh(
        #     ham.toarray(), k=1, which="SA", maxiter=1E4,)
        print(f'energy={e_min/l}')  # , f'energy_scipy={e_scipy/l}')
        # plt.plot(psi_0)
        # plt.plot(psi_scipy)
        # plt.show()

        zz = compute_correlation(psi_0, l=l, basis=basis, direction="xx")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="x")
        z = np.asarray(z)
        if r == 0:
            zs = z.reshape(1, -1)
            zzs = zz.reshape(1, zz.shape[0], zz.shape[1])
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            zzs = np.append(zzs, zz.reshape(
                1, zz.shape[0], zz.shape[1]), axis=0)

    return zs, zzs


def transverse_ising_sparse_h_k_mapping_check_lanczos_method(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    omega: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    j_nn = []
    for r in range(1, 5):
        if pbc:
            j_nn.append([[(j1/r)**6, i, (i + r) % l] for i in range(l)])  # pbc
        else:
            j_nn.append([[(j1/r)**6, i, (i + r) % l] for i in range(l)])  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        omega_h = [[omega, k] for k in range(l)]
        static = [["zz", j_nn[0]], ["zz", j_nn[1]], ["zz", j_nn[2]], ["zz", j_nn[3]], [
            "z", h], ["x", omega_h]]
        dynamic = []
        ham = hamiltonian(
            static,
            dynamic,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        # e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        e, psi_0 = lanczos_method(hamiltonian=ham, basis=basis)

        zz = compute_correlation(psi_0, l=l, basis=basis, direction="zz")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")
        z = np.asarray(z)
        if r == 0:
            zs = z.reshape(1, -1)
            zzs = zz.reshape(1, zz.shape[0], zz.shape[1])
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            zzs = np.append(zzs, zz.reshape(
                1, zz.shape[0], zz.shape[1]), axis=0)

    dir = "data/den2magn_dataset_1nn/"
    if check_2nn:
        dir = "data/den2magn_dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" +
        text_field + f"_{zzs.shape[0]}_n"
    )

    return zs, zzs
