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
# > He およびＨの核座標が， それぞれ (0, 0,0)， (0, 0, 1.4) （bohr単位） のとき，  HeH の基底状態 （1σ2 2σ1） に対する UHF波動関数とUHF エネルギーを求めよ． 基底  関数は STO-NG (N＝1, 2, 3, 4, 5, 6) のいずれかとして， 付録の HeH 系の分子積分  の値を用 い よ． SCF の収束判定は， エ ネ ル ギ ー に つ い て の み行い， 閾値は  0.0001 hartree とせよ． また， 得られた UHF波動関数を用いて Mulliken の電子密度  解析を行い， He およびＨ原子の電荷およびスピン密度を求めよ． 
# > 
# > 中井浩巳. 手で解く量子化学 I (p. 99). (Function). Kindle Edition. 

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
# 核電荷$Z_\mathrm{He}=2.0$, $Z_\mathrm{H}=1.0$, 核座標$\boldsymbol{R}_\mathrm{He}=(0.0, 0.0, 0.0)$, $\boldsymbol{R}_\mathrm{H}=(0.0, 0.0, 1.4)$より、核反発エネルギー

# %% [markdown]
# $$
# V_\mathrm{nuc} = \sum_{A=1}^M \sum_{B>A}^M \frac{Z_A Z_B}{R_{AB}}   \tag{3.2}
# $$

# %%
Z_He = 2.0
"""Heの核電荷"""
Z_H = 1.0
"""Hの核電荷"""

# %%
V_nuc = (Z_He * Z_H) / 1.4
"""核間のポテンシャルエネルギー"""
V_nuc

# %%
N_alpha = 2
"""upスピンの電子数"""
N_beta = 1
"""downスピンの電子数"""

# %% [markdown]
# ## 正準直行化の実行
#
# 重なり積分 $\boldsymbol{S}$ は、核座標と基底関数にのみ依存し、電子数には依存しない。

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
"""s: 固有値
U: 固有ベクトル"""
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
#
# 核座標と基底関数のみに依存し、電子数には依存しない。

# %%
H = np.array([[-2.6444, -1.5118], [-1.5118, -1.7782]])
"""コアハミルトニアン行列 (付録より)"""
H

# %% [markdown]
# ## 密度行列の初期値

# %% [markdown]
# ### 係数行列を求める方法

# %% [markdown]
# #### 1-4-1と同じ方法
#
# コアハミルトニアン行列を使うパターン

# %% [markdown]
# 直行化基底に対するコアハミルトン行列 $H'$ は、
#
# $$
# H'=X^T HX   \tag{4.2}
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
# #### MO軌道をAOと仮定する方法

# %% [markdown]
# $$
# \varphi_1^\alpha = \varphi_1^\beta = \chi_1 \\
# \varphi_2^\alpha = \varphi_2^\beta = \chi_2
# $$
#
# とする。
#
# つまり、1番目のMOはHeの1s軌道、2番目のMOはHの1s軌道と仮定している。
#
# これらは直交ではないので直交化してあげる必要がある。
#
# - 正準直交化: せっかく意味のあるMO軌道を仮定しているのに全く意味がなくなるので不適
# - 対称直交化: HF法では、占有軌道のみ直交化されれば良いのだが、すべての分子軌道に対して直交化が行われるので不適
#
# -> Gram-Schmidtの直交化 (今回のケースでは、$\alpha$電子のすべてのMOが$\alpha$電子の占有軌道となり、メリットはないが。)
#

# %% [markdown]
# 1番目の分子軌道について、
#
# $$
# |\varphi'_1 \rangle = \frac{|\varphi_1 \rangle}{\sqrt{\langle \varphi_1 |  \varphi_1 \rangle}} \tag{4.44}
# $$
#
# より、
#
# $$
# |\varphi'_1 \rangle = \frac{|\chi_1 \rangle}{\sqrt{\langle \chi_1 |  \chi_1 \rangle}} = \frac{|\chi_1 \rangle}{\sqrt{1.0}} = |\chi_1 \rangle
# $$

# %% [markdown]
# 2番目の分子軌道について、
#
# $$
# |\varphi''_2 \rangle = |\varphi_2 \rangle - |\varphi'_1 \rangle  \langle \varphi'_1 | \varphi_2 \rangle = \left( 1-|\varphi'_1 \rangle \langle \varphi'_1 | \right) | \varphi_2 \rangle \tag{4.45}
# $$
#
# より、
#
# $$
# \begin{align*}
#     \varphi''_2 &= |\varphi_2 \rangle - |\varphi'_1 \rangle  \langle \varphi'_1 | \varphi_2 \rangle \\
#     &= |\chi_2 \rangle - |\chi_1 \rangle \langle \chi_1 | \chi_2 \rangle \\
#     &= - 0.56059 \times |\chi_1 \rangle + |\chi_2 \rangle\ \ \ \because 付録より \langle \chi_1 | \chi_2 \rangle = 0.56059
# \end{align*}
# $$

# %% [markdown]
# さらに、規格化する。
#
# $$
# |\varphi'_2 \rangle = \frac{|\varphi''_2 \rangle}{\sqrt{\langle \varphi''_2 | \varphi''_2 \rangle}} \tag{4.46}
# $$
#
# より、
#
# $$
# \begin{align*}
#     |\varphi'_2 \rangle &= \frac{|\varphi''_2 \rangle}{\sqrt{\langle \varphi''_2 | \varphi''_2 \rangle}} \\
#     &= \frac{- 0.56059 \times |\chi_1 \rangle + |\chi_2 \rangle}{\sqrt{(- 0.56059)^2 \langle \chi_1 | \chi_1 \rangle + 2 \cdot (- 0.56059) \langle \chi_1 | \chi_2 \rangle + \langle \chi_2 | \chi_2 \rangle }} \\
#     &= \frac{- 0.56059 \times |\chi_1 \rangle + |\chi_2 \rangle}{\sqrt{(- 0.56059)^2 \cdot 1.0 + 2 \cdot (- 0.56059) \cdot (-0.56059) + 1.0 }} \\
#     &= \frac{- 0.56059 \times |\chi_1 \rangle + |\chi_2 \rangle}{1.39384} \\
#     &= -0.40220 \times |\chi_1 \rangle + 0.71744 \times |\chi_2 \rangle
# \end{align*}
#
#
# $$

# %% [markdown]
# つまり、

# %%
C = np.array([[1.0, -0.40220], [0.0, 0.71744]])
"""分子軌道係数行列"""
C


# %% [markdown]
# ### 係数行列から密度行列を求める

# %% [markdown]
# 密度行列 $\boldsymbol{P}$ は、$\alpha$スピン、$\beta$スピンそれぞれに対して、
#
# $$
# P_{\mu\nu}^\alpha = 2 \sum_{i} {c_{\mu i}^\alpha}^* c_{\nu i}^\alpha \ ,\  P_{\mu\nu}^\beta = 2 \sum_{i} {c_{\mu i}^\beta}^* c_{\nu i}^\beta \tag{5.19}
# $$
#
# $i$は、分子軌道の足、$\mu, \nu$は原子軌道の足

# %% [markdown]
# HeH では、$N_\alpha=2, N_\beta=1$より、
#
# $$
# \begin{align*}
#     P_{\mu \nu}^\alpha &= 2 \left(c_{\mu 1}^\alpha c_{\nu 1}^\alpha + c_{\mu 2}^\alpha c_{\nu 2}^\alpha \right) \\
#     P_{\mu \nu}^\beta &= 2 c_{\mu 1}^\beta c_{\nu 1}^\beta
# \end{align*}
# $$

# %%
def calculate_density_matrix_alpha(C: np.ndarray) -> np.ndarray:
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
    return C @ C.T  # 1-4-1と違って2を掛けないのは閉殻でないから。


P_alpha = calculate_density_matrix_alpha(C)
"""密度行列"""
P_alpha


# %%
def calculate_density_matrix_beta(C: np.ndarray) -> np.ndarray:
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
    return (
        C[:, [0]] @ C[:, [0]].T
    )  # 1-4-1と違って2を掛けないのは閉殻でないから。


P_beta = calculate_density_matrix_beta(C)
"""密度行列"""
P_beta

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
# F_{\mu \nu}^\alpha = H_{\mu \nu} + G_{\mu \nu}^\alpha  \\
# F_{\mu \nu}^\beta = H_{\mu \nu} + G_{\mu \nu}^\beta \tag{5.26}
# $$
#
# $H_{\mu \nu}$ は、[先ほど](#コアハミルトニアン行列の計算)求めているので、$G_{\mu \nu}^\alpha, G_{\mu \nu}^\beta$ を求めたい。
#
# $$
# \begin{align*}
#     J_{\mu \nu}^\alpha = \sum_{\lambda, \sigma} (\mu \nu | \lambda \sigma) P_{\lambda \sigma}^\alpha \ &,\ J_{\mu \nu}^\beta = \sum_{\lambda, \sigma} (\mu \nu | \lambda \sigma) P_{\lambda \sigma}^\beta \tag{5.21} \\
#     K_{\mu \nu}^\alpha = \sum_{\lambda, \sigma} (\mu \sigma | \lambda \nu) P_{\lambda \sigma}^\alpha \ &,\ K_{\mu \nu}^\beta = \sum_{\lambda, \sigma} (\mu \sigma | \lambda \nu) P_{\lambda \sigma}^\beta \tag{5.22} \\
#     G_{\mu \nu}^\alpha = J_{\mu \nu}^\alpha - K_{\mu \nu}^\alpha + J_{\mu \nu}^\beta \ &,\ G_{\mu \nu}^\beta = J_{\mu \nu}^\beta - K_{\mu \nu}^\beta + J_{\mu \nu}^\alpha \tag{5.23} \\
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


# %%
J_alpha = calculate_coulomb_energy_matrix(P_alpha)
"""クーロンエネルギー行列"""
J_alpha

# %%
J_beta = calculate_coulomb_energy_matrix(P_beta)
"""クーロンエネルギー行列"""
J_beta


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


# %%
K_alpha = calculate_exchange_energy_matrix(P_alpha)
"""交換エネルギー行列"""
K_alpha

# %%
K_beta = calculate_exchange_energy_matrix(P_beta)
"""交換エネルギー行列"""
K_beta

# %%
G_alpha = J_alpha - K_alpha + J_beta
"""2電子積分行列"""
G_alpha

# %%
G_beta = J_beta - K_beta + J_alpha
"""2電子積分行列"""
G_beta

# %%
F_alpha = H + G_alpha
"""Fock行列"""
F_alpha

# %%
F_beta = H + G_beta
"""Fock行列"""
F_beta


# %%
def density2fock(
    P_alpha: np.ndarray, P_beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """密度行列からFock行列を計算する関数

    Parameters
    ----------
    P_alpha : np.ndarray
        upスピンの密度行列
    P_beta : np.ndarray
        downスピンの密度行列

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        upスピンとdownスピンのFock行列
    """
    J_alpha = calculate_coulomb_energy_matrix(P_alpha)
    K_alpha = calculate_exchange_energy_matrix(P_alpha)
    J_beta = calculate_coulomb_energy_matrix(P_beta)
    K_beta = calculate_exchange_energy_matrix(P_beta)

    G_alpha = J_alpha - K_alpha + J_beta
    G_beta = J_beta - K_beta + J_alpha
    return H + G_alpha, H + G_beta


# %%
density2fock(P_alpha, P_beta)

# %% [markdown]
# ## RHFエネルギーの計算

# %% [markdown]
# $$
# E_0^\mathrm{UHF} = \frac{1}{2} \sum_{\mu,\nu} P_{\mu \nu}^\alpha \left[ H_{\mu \nu} + F_{\mu \nu}^\alpha \right] + \frac{1}{2} \sum_{\mu,\nu} P_{\mu \nu}^\beta \left[ H_{\mu \nu} + F_{\mu \nu}^\beta \right]  \tag{5.27}
# $$

# %%
E_0 = 0.5 * np.sum(P_alpha * (H + F_alpha)) + 0.5 * np.sum(
    P_beta * (H + F_beta)
)
"""ハミルトニアンに対するRHFエネルギー"""
E_0

# %%
E_tot = E_0 + V_nuc
"""RHF全エネルギー"""
E_tot


# %%
def calculate_total_energy(P_alpha: np.ndarray, P_beta: np.ndarray) -> float:
    """全エネルギーを計算する関数

    Parameters
    ----------
    P_alpha : np.ndarray
        upスピンの密度行列
    P_beta : np.ndarray
        downスピンの密度行列

    Returns
    -------
    float
        全エネルギー
    """
    F_alpha, F_beta = density2fock(P_alpha, P_beta)
    return (
        0.5 * np.sum(P_alpha * (H + F_alpha))
        + 0.5 * np.sum(P_beta * (H + F_beta))
        + V_nuc
    )


# %%
calculate_total_energy(P_alpha, P_beta)


# %% [markdown]
# ## Pople-Nesbet方程式の解法
#
# 基本的には[Fock行列を求める](#fock行列の計算)から[RHFエネルギーの計算](#rhfエネルギーの計算)までを繰り返して、収束要件を満たしたら終わり。

# %%
def scf_cycle(
    P_alpha: np.ndarray, P_beta: np.ndarray, threshold: float = 1e-4
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """SCFサイクルを行う関数

    Parameters
    ----------
    P_alpha : np.ndarray
        upスピンの密度行列
    P_beta : np.ndarray
        downスピンの密度行列

    Returns
    -------
    tuple[float, tuple[np.ndarray, np.ndarray]]
        全エネルギーとupスピン、downスピンの密度行列
    """
    # 初期化
    E_tot_prev = np.inf
    # 全エネルギー
    E_tot = calculate_total_energy(P_alpha, P_beta)
    # 収束判定 (全エネルギーの変化量が閾値以下になるまで繰り返す)
    while (E_tot_prev - E_tot) > threshold:
        print(f"{E_tot=}")
        # 全エネルギーを保存
        E_tot_prev = E_tot
        # 密度行列からFock行列を計算
        F_alpha, F_beta = density2fock(P_alpha, P_beta)
        # 直交化基底に対するFock行列
        F_alpha_prime = X.T @ F_alpha @ X
        F_beta_prime = X.T @ F_beta @ X
        # Fock行列をユニタリ対角化して、(軌道エネルギーと) 分子軌道係数行列を求める
        _, C_alpha_prime = unitary_diagonalization(F_alpha_prime)
        _, C_beta_prime = unitary_diagonalization(F_beta_prime)
        # 分子軌道係数行列を元の基底に戻す
        C_alpha = X @ C_alpha_prime
        C_beta = X @ C_beta_prime
        # 密度行列を計算
        P_alpha = calculate_density_matrix_alpha(C_alpha)
        P_beta = calculate_density_matrix_beta(C_beta)
        # 全エネルギーを計算
        E_tot = calculate_total_energy(P_alpha, P_beta)
    return E_tot, (P_alpha, P_beta)


# %%
energy_total_opt, (P_alpha_opt, P_beta_opt) = scf_cycle(P_alpha, P_beta)
"""最適化された全エネルギーと密度行列"""
print(f"{energy_total_opt=}")
print(f"{P_alpha_opt=}")
print(f"{P_beta_opt=}")

# %% [markdown]
# ## 分子物性の計算

# %% [markdown]
# Mullikenの電荷解析
#
# $$
# N_A^\mathrm{Mul} = \sum_{\mu \in A} \sum_{\nu} P_{\mu \nu} S_{\nu \mu} = \sum_{\mu \in A} (\rm{PS})_{\mu \mu} \tag{4.53}
# $$

# %%
PS_alpha = P_alpha_opt @ S
"""upスピンの密度行列と重なり行列の積"""
PS_beta = P_beta_opt @ S
"""downスピンの密度行列と重なり行列の積"""

N_H_alpha = PS_alpha[0, 0]
"""upスピンのH原子の電子数"""
N_H_beta = PS_beta[0, 0]
"""downスピンのH原子の電子数"""
N_He_alpha = PS_alpha[1, 1]
"""upスピンのHe原子の電子数"""
N_He_beta = PS_beta[1, 1]
"""downスピンのHe原子の電子数"""

N_H = N_H_alpha + N_H_beta
"""H原子の電子数"""
N_He = N_He_alpha + N_He_beta
"""He原子の電子数"""

print(f"{N_H=}")
print(f"{N_He=}")

# %% [markdown]
# よって、Mulliken電荷は、
# $$
# Q_A = Z_A - N_A^\mathrm{mul} \tag{4.54}
# $$
# より、

# %%
Q_He = Z_He - N_He
"""Heの電荷"""
Q_H = Z_H - N_H
"""Hの電荷"""
print(f"{Q_He=}")
print(f"{Q_H=}")

# %%
F_alpha_opt, F_beta_opt = density2fock(P_alpha_opt, P_beta_opt)

epsilon_alpha_opt, C_prime_alpha_opt = unitary_diagonalization(
    X.T @ F_alpha_opt @ X
)
epsilon_beta_opt, C_prime_beta_opt = unitary_diagonalization(
    X.T @ F_beta_opt @ X
)

C_alpha_opt = X @ C_prime_alpha_opt
C_beta_opt = X @ C_prime_beta_opt

print(f"{epsilon_alpha_opt=}")
print(f"{C_alpha_opt=}")
print(f"{epsilon_beta_opt=}")
print(f"{C_beta_opt=}")

# %% [markdown]
# Koopmansの定理より、イオン化エネルギーおよび電子親和力は、

# %%
ionic_potential_energy = -epsilon_alpha_opt[1, 1]
"""イオン化ポテンシャルエネルギー"""
print(f"{ionic_potential_energy=} hartree")

# %%
electron_affinity = -epsilon_beta_opt[1, 1]
"""電気陰性度"""
print(f"{electron_affinity=} hartree")
