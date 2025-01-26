# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %% [markdown]
# > He およびＨの核座標がそれぞれ (0, 0, 0)， (0, 0, 1.4) （bohr単位） のとき， HeH＋  の基底状態 （1σ2） に対する RHF波動関数とRHF エネルギーを求めよ． 基底関数は  STO-NG (N＝1, 2, 3, 4, 5, 6) のいずれかとして， 付録の HeH系の分子積分の値を用  いよ． SCF の収束判定は， エネルギーについてのみ行い， 閾値は 0.0001 hartree とせ  よ． また， 得られた RHF波動関数を用いて Mulliken の電子密度解析を行い， He お  よびＨ原子の電荷を求めよ． 
# >
# > 中井浩巳. 手で解く量子化学 I (p. 86). (Function). Kindle Edition. 

# %% [markdown]
# 基底関数は、STO-6Gとする。

# %%
from types import MappingProxyType
from itertools import product

import numpy as np
from scipy.linalg import ishermitian

# %% [markdown]
# ## 対象分子の設定

# %% [markdown]
# 核電荷$Z_\mathrm{He}=2.0$, $Z_\mathrm{H}=1.0$, 核座標$\bm{R_\mathrm{He}}=(0.0, 0.0, 0.0)$, $\bm{R_\mathrm{H}}=(0.0, 0.0, 1.4)$より、核反発エネルギー

# %% [markdown]
# $$
# V_\mathrm{nuc} = \sum_{A=1}^M \sum_{B>A}^M \frac{Z_A Z_B}{R_{AB}}   \tag{3.2}
# $$

# %%
V_nuc = (2.0 * 1.0) / 1.4
"""核間のポテンシャルエネルギー"""
V_nuc

# %% [markdown]
# ## 正準直行化の実行

# %%
S = np.array([[1.0, 0.56059], [0.56059, 1.0]])
"""重なり行列 (付録より)"""
S


# %%
def unitary_diagonalization(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ユニタリ対角化関数

    Parameters
    ----------
    arr : np.ndarray
        対角化する行列

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        対角化された行列とユニタリ行列

    Raises
    ------
    ValueError
        エルミート行列でない場合
    """
    if ishermitian(arr, rtol=1e-05, atol=1e-08):
        # エルミート行列の固有ベクトル
        _, u = np.linalg.eigh(arr)

        # Uの随伴行列(共役転置)
        u_dagger = np.conjugate(u.T)

        # エルミート行列をユニタリ行列で対角化
        diag_matrix = u_dagger @ arr @ u

        # データ型を整数に変更して小さな誤差が入っている虚部を取り除く
        # 単位行列をかけて対角成分以外の要素の誤差を小さくする
        diag_matrix = diag_matrix.astype(np.float64) * np.identity(
            arr.shape[0], dtype=np.float64
        )

        return diag_matrix, u
    else:
        raise ValueError("The matrix is not Hermitian.")


s, U = unitary_diagonalization(S)
print(f"{s=}")
print(f"{U=}")


# %% [markdown]
# 正準直交化の変換行列 $X$ は、
#
# $$
# X_\mathrm{can} = U s^{-\frac{1}{2}} \tag{4.36}
# $$

# %%
# 行列を^(-1/2)する。'-' は逆行列の意味
def inverse_sqrt_matrix(arr: np.ndarray) -> np.ndarray:
    """行列の^(-1/2)を計算する関数

    Parameters
    ----------
    arr : np.ndarray
        ^(-1/2)を計算する行列

    Returns
    -------
    np.ndarray
        ^(-1/2)された行列
    """
    return np.linalg.inv(np.sqrt(arr))


inverse_sqrt_matrix(s)

# %%
X = U @ inverse_sqrt_matrix(s)
"""正準直行化の変換行列"""
X

# %% [markdown]
# ## コアハミルトニアン行列の計算

# %%
H = np.array([[-2.6444, -1.5118], [-1.5118, -1.7782]])
"""コアハミルトニアン行列 (付録より)"""
H

# %% [markdown]
# ## 密度行列の初期値

# %% [markdown]
# 直行化基底に対するコアハミルトン行列 $H'$ は、
#
# $$
# H'=X^T HX
# $$

# %%
H_prime = X.T @ H @ X
"""正準直交化基底に対するコアハミルトニアン行列"""
H_prime

# %%
_, C_prime = unitary_diagonalization(H_prime)
"""正準直交化基底に対する分子軌道係数行列"""
C_prime

# %%
C = X @ C_prime
"""分子軌道係数行列"""
C


# %% [markdown]
# 密度行列 $\bm{P}$ は、
#
# $$
# P_{\mu\nu} = 2 \sum_{i} c_{\mu i}^* c_{\nu i} \tag{4.19}
# $$
#
# $i$は、分子軌道の足、$\mu, \nu$は原子軌道の足

# %% [markdown]
# HeH+ では、占有軌道 $\varphi_i$ だけなので、 ($1\sigma$ 軌道だけという意味と解釈)

# %%
def calculate_density_matrix(C: np.ndarray) -> np.ndarray:
    """密度行列を計算する関数

    Parameters
    ----------
    C : np.ndarray
        分子軌道係数行列

    Returns
    -------
    np.ndarray
        密度行列
    """
    return 2 * (C[:, [0]] @ C[:, [0]].T)


P = calculate_density_matrix(C)
"""密度行列"""
P

# %% [markdown]
# ## 電子反発積分の計算

# %%
MAP_ELECTRON_REPULSION_INTEGRAL = MappingProxyType(
    {
        ((1, 1), (1, 1)): 1.05625,
        ((1, 1), (1, 2)): 0.46768,
        ((1, 1), (2, 1)): 0.46768,
        ((1, 1), (2, 2)): 0.60640,
        ((1, 2), (1, 1)): 0.46768,
        ((1, 2), (1, 2)): 0.24649,
        ((1, 2), (2, 1)): 0.24649,
        ((1, 2), (2, 2)): 0.38871,
        ((2, 1), (1, 1)): 0.46768,
        ((2, 1), (1, 2)): 0.24649,
        ((2, 1), (2, 1)): 0.24649,
        ((2, 1), (2, 2)): 0.38871,
        ((2, 2), (1, 1)): 0.60640,
        ((2, 2), (1, 2)): 0.38871,
        ((2, 2), (2, 1)): 0.38871,
        ((2, 2), (2, 2)): 0.77500,
    }
)
"""電子反発積分 (付録より)"""
None


# %% [markdown]
# ## Fock行列の計算

# %% [markdown]
# $$
# F_{\mu \nu} = H_{\mu \nu} + G_{\mu \nu} \tag{4.3} \\
# $$
#
# $H_{\mu \nu}$ は、[先ほど](#コアハミルトニアン行列の計算)求めているので、$G_{\mu \nu}$ を求めたい。
#
# $$
# \begin{align*}
#     J_{\mu \nu} &= \sum_{\lambda, \sigma} (\mu \nu | \lambda \sigma) P_{\lambda \sigma} \tag{4.25} \\
#     K_{\mu \nu} &= \sum_{\lambda, \sigma} (\mu \sigma | \lambda \nu) P_{\lambda \sigma} \tag{4.26} \\
#     G_{\mu \nu} &= J_{\mu \nu} - \frac{1}{2} K_{\mu \nu} \tag{4.27} \\
# \end{align*}
# $$

# %%
def calculate_coulomb_energy_matrix(P: np.ndarray) -> np.ndarray:
    """クーロンエネルギー行列を計算する関数

    Parameters
    ----------
    P : np.ndarray
        密度行列

    Returns
    -------
    np.ndarray
        クーロンエネルギー行列
    """
    return np.array(
        [
            [
                sum(
                    MAP_ELECTRON_REPULSION_INTEGRAL[
                        (_mu, _nu), (_lambda, _sigma)
                    ]
                    * P[_lambda - 1, _sigma - 1]  # Pは0-indexedなので-1する
                    for _lambda, _sigma in product((1, 2), repeat=2)
                )
                for _nu in (1, 2)
            ]
            for _mu in (1, 2)
        ]
    )


J = calculate_coulomb_energy_matrix(P)
"""クーロンエネルギー行列"""
J


# %%
def calculate_exchange_energy_matrix(P: np.ndarray) -> np.ndarray:
    """交換エネルギー行列を計算する関数

    Parameters
    ----------
    P : np.ndarray
        密度行列

    Returns
    -------
    np.ndarray
        交換エネルギー行列
    """
    return np.array(
        [
            [
                sum(
                    MAP_ELECTRON_REPULSION_INTEGRAL[
                        (_mu, _sigma), (_lambda, _nu)
                    ]
                    * P[_lambda - 1, _sigma - 1]  # Pは0-indexedなので-1する
                    for _lambda, _sigma in product((1, 2), repeat=2)
                )
                for _nu in (1, 2)
            ]
            for _mu in (1, 2)
        ]
    )


K = calculate_exchange_energy_matrix(P)
"""交換エネルギー行列"""
K

# %%
G = J - 0.5 * K
"""2電子積分行列"""
G

# %%
F = H + G
"""Fock行列"""
F


# %%
def density2fock(P: np.ndarray) -> np.ndarray:
    """密度行列からFock行列を計算する関数

    Parameters
    ----------
    P : np.ndarray
        密度行列

    Returns
    -------
    np.ndarray
        Fock行列
    """
    G = calculate_coulomb_energy_matrix(
        P
    ) - 0.5 * calculate_exchange_energy_matrix(P)
    return H + G


density2fock(P)

# %% [markdown]
# ## RHFエネルギーの計算

# %%
E_0 = 0.5 * np.sum(P * (H + F))
"""ハミルトニアンに対するRHFエネルギー"""
E_0

# %%
E_tot = E_0 + V_nuc
"""RHF全エネルギー"""
E_tot


# %%
def calculate_total_energy(P: np.ndarray) -> float:
    """全エネルギーを計算する関数

    Parameters
    ----------
    P : np.ndarray
        密度行列

    Returns
    -------
    float
        全エネルギー
    """
    F = density2fock(P)
    return 0.5 * np.sum(P * (H + F)) + V_nuc


calculate_total_energy(P)


# %% [markdown]
# ## Roothan-hall方程式の解法
#
# 基本的には[Fock行列を求める](#fock行列の計算)から[RHFエネルギーの計算](#rhfエネルギーの計算)までを繰り返して、収束要件を満たしたら終わり。

# %%
def scf_cycle(
    P: np.ndarray, threshold: float = 1e-4
) -> tuple[float, np.ndarray]:
    """SCFサイクルを行う関数

    Parameters
    ----------
    P : np.ndarray
        密度行列

    Returns
    -------
    tuple[float, np.ndarray]
        全エネルギーと密度行列
    """
    # 初期化
    E_tot_prev = np.inf
    # 全エネルギー
    E_tot = calculate_total_energy(P)
    # 収束判定 (全エネルギーの変化量が閾値以下になるまで繰り返す)
    while (E_tot_prev - E_tot) > threshold:
        print(f"{E_tot=}")
        # 全エネルギーを保存
        E_tot_prev = E_tot
        # 密度行列からFock行列を計算
        F = density2fock(P)
        # 直交化基底に対するFock行列
        F_prime = X.T @ F @ X
        # Fock行列をユニタリ対角化して、(軌道エネルギーと) 分子軌道係数行列を求める
        _, C_prime = unitary_diagonalization(F_prime)
        # 分子軌道係数行列を元の基底に戻す
        C = X @ C_prime
        # 密度行列を計算
        P = calculate_density_matrix(C)
        # 全エネルギーを計算
        E_tot = calculate_total_energy(P)
    return E_tot, P


# %%
energy_total_opt, P_opt = scf_cycle(P)
print(f"{energy_total_opt=}")
print(f"{P_opt=}")

# %% [markdown]
# ## 分子物性の計算

# %% [markdown]
# Mullikenの電荷解析
#
# $$
# N_A^\mathrm{Mul} = \sum_{\mu \in A} \sum_{\nu} P_{\mu \nu} S_{\nu \mu} = \sum_{\mu \in A} (\rm{PS})_{\mu \mu} \tag{4.53}
# $$

# %%
N_He = (P_opt @ S)[0, 0]
"""Heの電子数"""
N_H = (P_opt @ S)[1, 1]
"""Hの電子数"""
print(f"{N_He=}")
print(f"{N_H=}")

# %% [markdown]
# よって、Mulliken電荷は、
# $$
# Q_A = Z_A - N_A^\mathrm{mul} \tag{4.54}
# $$
# より、

# %%
Q_He = 2 - N_He
"""Heの電荷"""
Q_H = 1 - N_H
"""Hの電荷"""
print(f"{Q_He=}")
print(f"{Q_H=}")

# %%
epsilon_opt, C_prime_opt = unitary_diagonalization(
    X.T @ density2fock(P_opt) @ X
)
C_opt = X @ C_prime_opt
print(f"{epsilon_opt=}")
print(f"{C_opt=}")
