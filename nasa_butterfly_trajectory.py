"""
mini_interplanetary_transfer_optionA.py
--------------------------------------
Mini interplanetary trajectory design (NASA-style pipeline, simplified):

1) Planet ephemeris (planar circular heliocentric orbits for Earth/Mars)
2) Lambert solver (universal variable method)
3) Porkchop plot: total DV vs departure date and time-of-flight
4) Find best DV window
5) Sensitivity / "butterfly-style": arrival miss distance vs small departure time shift
6) Uncertainty propagation in (t0, TOF):
   - Monte Carlo
   - Tensor Gauss–Hermite quadrature (Gaussian uncertainty)

Outputs:
  - porkchop.png
  - best_trajectory.png
  - sensitivity_miss_distance.png
  - uq_convergence_mean.png, uq_convergence_var.png
  - uq_hist_mc_vs_quad.png

No external dependencies beyond numpy/matplotlib.

Units:
  - Distance: AU
  - Time: days
  - mu_sun in AU^3/day^2
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# =========================
# 0) CONSTANTS & SETTINGS
# =========================
OUTDIR = "INTERPLANETARY_FIGS"
os.makedirs(OUTDIR, exist_ok=True)

rng = np.random.default_rng(0)

# Sun gravitational parameter in AU^3/day^2:
# mu = (2*pi / period)^2 * a^3; for Earth: a=1 AU, period=365.25 days
MU_SUN = (2.0 * np.pi / 365.25) ** 2

# Circular orbit radii (AU) and periods (days)
A_EARTH = 1.0
A_MARS = 1.523679
PER_EARTH = 365.25
PER_MARS = 686.98

# Mean motions (rad/day)
N_EARTH = 2.0 * np.pi / PER_EARTH
N_MARS = 2.0 * np.pi / PER_MARS

# Phase offset for Mars (to create a non-trivial window)
MARS_PHASE0 = np.deg2rad(40.0)  # tweak if you want


# =========================
# 1) PLANET STATE (circular, coplanar)
# =========================
def planet_state_circular(a: float, n: float, t: float, phase0: float = 0.0):
    """
    Heliocentric planar circular orbit state in AU, AU/day.
    r = a [cos(theta), sin(theta)]
    v = a*n [-sin(theta), cos(theta)]
    """
    theta = n * t + phase0
    r = a * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    v = a * n * np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    return r, v


def earth_state(t: float):
    return planet_state_circular(A_EARTH, N_EARTH, t, phase0=0.0)


def mars_state(t: float):
    return planet_state_circular(A_MARS, N_MARS, t, phase0=MARS_PHASE0)


# =========================
# 2) STUMPFF FUNCTIONS (Lambert universal variable)
# =========================
def stumpff_C(z: float) -> float:
    if z > 1e-8:
        s = np.sqrt(z)
        return (1.0 - np.cos(s)) / z
    if z < -1e-8:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1.0) / (-z)
    # series
    return 0.5 - z / 24.0 + z**2 / 720.0


def stumpff_S(z: float) -> float:
    if z > 1e-8:
        s = np.sqrt(z)
        return (s - np.sin(s)) / (s**3)
    if z < -1e-8:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / (s**3)
    # series
    return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0


# =========================
# 3) LAMBERT SOLVER (universal variable, short-way)
# =========================
def lambert_universal(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float, prograde: bool = True,
                      max_iter: int = 100, tol: float = 1e-10):
    """
    Solve Lambert's problem from r1 to r2 in time dt using universal variables.
    Returns (v1, v2) transfer velocities at r1 and r2.

    Notes:
      - Planar 2D version (works fine)
      - Uses "short way" by default; prograde controls sign of transfer angle.
    """
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)

    # Transfer angle
    cross_z = r1[0] * r2[1] - r1[1] * r2[0]
    dot = float(np.dot(r1, r2))
    cos_dtheta = np.clip(dot / (r1n * r2n), -1.0, 1.0)
    dtheta = np.arccos(cos_dtheta)

    # Choose prograde/retrograde by sign of cross product
    if prograde:
        if cross_z < 0:
            dtheta = 2.0 * np.pi - dtheta
    else:
        if cross_z >= 0:
            dtheta = 2.0 * np.pi - dtheta

    A = np.sin(dtheta) * np.sqrt(r1n * r2n / (1.0 - np.cos(dtheta)))
    if abs(A) < 1e-14:
        raise RuntimeError("Lambert solver failed: A ~ 0 (degenerate geometry).")

    # Root-find on z
    z = 0.0
    z_low, z_high = -4.0 * np.pi**2, 4.0 * np.pi**2

    def time_of_flight(zval: float) -> float:
        C = stumpff_C(zval)
        S = stumpff_S(zval)
        y = r1n + r2n + A * (zval * S - 1.0) / np.sqrt(C)
        if y < 0:
            return np.inf
        x = np.sqrt(y / C)
        return (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)

    # Ensure brackets produce finite values
    for _ in range(60):
        if np.isfinite(time_of_flight(z_low)) and np.isfinite(time_of_flight(z_high)):
            break
        z_low *= 0.8
        z_high *= 0.8

    # Newton with safeguarding (bisection fallback)
    for _ in range(max_iter):
        t_z = time_of_flight(z)
        if not np.isfinite(t_z):
            z = 0.5 * (z + z_high)
            continue

        F = t_z - dt
        if abs(F) < tol:
            break

        # Numerical derivative dF/dz (robust)
        dz = 1e-6
        t1 = time_of_flight(z + dz)
        t0 = time_of_flight(z - dz)
        dF = (t1 - t0) / (2.0 * dz)

        # If derivative is bad, bisect
        if not np.isfinite(dF) or abs(dF) < 1e-12:
            if F > 0:
                z_high = z
            else:
                z_low = z
            z = 0.5 * (z_low + z_high)
            continue

        z_new = z - F / dF

        # Safeguard
        if z_new < z_low or z_new > z_high or not np.isfinite(z_new):
            if F > 0:
                z_high = z
            else:
                z_low = z
            z = 0.5 * (z_low + z_high)
        else:
            z = z_new

    # Compute f,g from final z
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1n + r2n + A * (z * S - 1.0) / np.sqrt(C)
    if y < 0:
        raise RuntimeError("Lambert solver failed: y < 0.")

    f = 1.0 - y / r1n
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2n

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


# =========================
# 4) TRANSFER COST (DV) + ARRIVAL MISS METRIC
# =========================
def transfer_dv_and_miss(t0: float, tof: float, prograde: bool = True):
    """
    Compute:
      - DV_total = |v1_trans - vE| + |vM - v2_trans|
      - miss distance if you propagate *ballistic* from departure using v1_trans for tof
        and compare to Mars position at arrival.

    This "miss" gives a sensitivity measure (how hard navigation errors matter).
    """
    t1 = t0 + tof
    rE, vE = earth_state(t0)
    rM, vM = mars_state(t1)

    v1t, v2t = lambert_universal(rE, rM, tof, MU_SUN, prograde=prograde)

    dv_dep = np.linalg.norm(v1t - vE)
    dv_arr = np.linalg.norm(vM - v2t)
    dv_total = dv_dep + dv_arr

    # Propagate the transfer with 2-body f,g over tof from rE,v1t and compare to rM
    # We can reuse universal variables propagation:
    r_pred = two_body_propagate_r(rE, v1t, tof, MU_SUN)
    miss = np.linalg.norm(r_pred - rM)

    return dv_total, dv_dep, dv_arr, miss


def two_body_propagate_r(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float):
    """
    Universal variable 2-body propagation (return position only).
    This is used to define a 'miss distance' diagnostic for sensitivity.
    """
    r0n = np.linalg.norm(r0)
    v0n = np.linalg.norm(v0)
    vr0 = float(np.dot(r0, v0) / r0n)
    alpha = 2.0 / r0n - (v0n**2) / mu

    # Initial guess for chi
    if abs(alpha) > 1e-12:
        chi = np.sqrt(mu) * abs(alpha) * dt
    else:
        # near-parabolic guess
        chi = np.sqrt(mu) * dt / r0n

    def F(chi_val: float) -> float:
        z = alpha * chi_val**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (r0n * vr0 / np.sqrt(mu)) * chi_val**2 * C + (1.0 - alpha * r0n) * chi_val**3 * S + r0n * chi_val - np.sqrt(mu) * dt

    # Newton iterations
    for _ in range(50):
        f = F(chi)
        if abs(f) < 1e-12:
            break
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        dF = (r0n * vr0 / np.sqrt(mu)) * chi * (1.0 - z * S) + (1.0 - alpha * r0n) * chi**2 * C + r0n
        if abs(dF) < 1e-14:
            break
        chi -= f / dF

    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    f_g = 1.0 - (chi**2 / r0n) * C
    g_g = dt - (chi**3 / np.sqrt(mu)) * S
    r = f_g * r0 + g_g * v0
    return r


# =========================
# 5) PORKCHOP GRID SEARCH
# =========================
def porkchop_scan(t0_grid, tof_grid):
    DV = np.full((len(tof_grid), len(t0_grid)), np.nan, dtype=float)
    MISS = np.full_like(DV, np.nan)

    for j, tof in enumerate(tof_grid):
        for i, t0 in enumerate(t0_grid):
            try:
                dv, _, _, miss = transfer_dv_and_miss(t0, tof, prograde=True)
                DV[j, i] = dv
                MISS[j, i] = miss
            except Exception:
                continue
    return DV, MISS


def find_best(DV, t0_grid, tof_grid):
    mask = np.isfinite(DV)
    idx = np.argmin(DV[mask])
    flat_indices = np.argwhere(mask)
    j, i = flat_indices[idx]
    return float(t0_grid[i]), float(tof_grid[j]), float(DV[j, i])


# =========================
# 6) GAUSS–HERMITE TENSOR QUADRATURE FOR UQ
# =========================
def gh_nodes_weights(n):
    """
    For xi ~ N(0,1), we want E[g(xi)] = 1/sqrt(2π) ∫ g(x) exp(-x^2/2) dx
    We'll use probabilists' hermegauss: nodes/weights for ∫ f(x) exp(-x^2/2) dx
    so E[g] ≈ (1/sqrt(2π)) Σ w_i g(x_i)
    """
    from numpy.polynomial.hermite_e import hermegauss
    x, w = hermegauss(n)
    wE = w / np.sqrt(2.0 * np.pi)
    return x, wE


def uq_tensor_gh(t0_mean, tof_mean, sig_t0, sig_tof, n):
    """
    Compute mean/var of DV(t0,tof) under Gaussian uncertainty:
      t0 = t0_mean + sig_t0 * xi
      tof = tof_mean + sig_tof * eta
    using n x n tensor Gauss-Hermite.
    """
    xi, wi = gh_nodes_weights(n)
    eta, wj = gh_nodes_weights(n)

    vals = []
    weights = []

    for a, wa in zip(xi, wi):
        for b, wb in zip(eta, wj):
            t0 = t0_mean + sig_t0 * a
            tof = tof_mean + sig_tof * b
            try:
                dv, _, _, _ = transfer_dv_and_miss(t0, tof, prograde=True)
            except Exception:
                dv = np.nan
            if np.isfinite(dv):
                vals.append(dv)
                weights.append(wa * wb)

    wsum = float(np.sum(weights))
    if wsum <= 0:
        return np.nan, np.nan

    weights = np.array(weights) / wsum
    vals = np.array(vals)
    mean = float(np.sum(weights * vals))
    var = float(np.sum(weights * (vals - mean) ** 2))
    return mean, var


# =========================
# 7) PLOTS
# =========================
def plot_porkchop(DV, t0_grid, tof_grid, best):
    plt.figure(figsize=(9, 6))
    # mask for plotting
    V = np.copy(DV)
    finite = np.isfinite(V)
    if np.any(finite):
        vmax = np.nanpercentile(V[finite], 70)
        vmin = np.nanpercentile(V[finite], 5)
        plt.imshow(V, origin="lower",
                   extent=[t0_grid[0], t0_grid[-1], tof_grid[0], tof_grid[-1]],
                   aspect="auto")
        plt.clim(vmin, vmax)

    t0b, tofb, dvb = best
    plt.scatter([t0b], [tofb], marker="x", s=120)
    plt.colorbar(label=r"Total $\Delta v$ (AU/day)")
    plt.xlabel("Departure time t0 (days from epoch)")
    plt.ylabel("Time of Flight TOF (days)")
    plt.title(r"Porkchop plot: $\Delta v(t_0,\mathrm{TOF})$ (simplified Earth→Mars)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "porkchop.png"), dpi=300)
    plt.close()


def plot_best_trajectory(t0_best, tof_best):
    t1 = t0_best + tof_best
    rE0, vE0 = earth_state(t0_best)
    rM1, vM1 = mars_state(t1)
    v1t, v2t = lambert_universal(rE0, rM1, tof_best, MU_SUN, prograde=True)

    # sample transfer arc by propagating
    ts = np.linspace(0, tof_best, 250)
    traj = np.array([two_body_propagate_r(rE0, v1t, dt, MU_SUN) for dt in ts])

    # sample orbits for plotting
    ts_orb = np.linspace(t0_best, t0_best + 800, 800)
    earth_orb = np.array([earth_state(t)[0] for t in ts_orb])
    mars_orb = np.array([mars_state(t)[0] for t in ts_orb])

    plt.figure(figsize=(7, 7))
    plt.plot(earth_orb[:, 0], earth_orb[:, 1], label="Earth orbit")
    plt.plot(mars_orb[:, 0], mars_orb[:, 1], label="Mars orbit")
    plt.plot(traj[:, 0], traj[:, 1], label="Transfer arc")
    plt.scatter([0], [0], s=50, label="Sun")
    plt.scatter([rE0[0]], [rE0[1]], label="Departure (Earth)")
    plt.scatter([rM1[0]], [rM1[1]], label="Arrival (Mars)")
    plt.axis("equal")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title("Best transfer arc (Lambert, simplified)")
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "best_trajectory.png"), dpi=300)
    plt.close()


def plot_sensitivity_miss(t0_best, tof_best):
    # "butterfly-style": tiny departure shift -> arrival miss change
    deltas = np.linspace(-3.0, 3.0, 61)  # days
    miss = []
    for d in deltas:
        try:
            _, _, _, m = transfer_dv_and_miss(t0_best + d, tof_best, prograde=True)
        except Exception:
            m = np.nan
        miss.append(m)
    miss = np.array(miss)

    plt.figure()
    plt.plot(deltas, miss)
    plt.xlabel("Departure time perturbation Δt0 (days)")
    plt.ylabel("Arrival miss distance (AU)")
    plt.title("Sensitivity: miss distance vs small departure-time perturbation")
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "sensitivity_miss_distance.png"), dpi=300)
    plt.close()


def plot_uq_mc_vs_quad(t0_best, tof_best, sig_t0, sig_tof):
    # Monte Carlo "truth"
    N = 40000
    xi = rng.standard_normal(N)
    eta = rng.standard_normal(N)
    t0_s = t0_best + sig_t0 * xi
    tof_s = tof_best + sig_tof * eta

    dv_mc = []
    for t0, tof in zip(t0_s, tof_s):
        try:
            dv, _, _, _ = transfer_dv_and_miss(float(t0), float(tof), prograde=True)
            dv_mc.append(dv)
        except Exception:
            continue
    dv_mc = np.array(dv_mc)

    # GH tensor quadrature at a decent n
    mean_q, var_q = uq_tensor_gh(t0_best, tof_best, sig_t0, sig_tof, n=9)

    plt.figure()
    plt.hist(dv_mc, bins=60, density=True, alpha=0.7, label="Monte Carlo")
    plt.axvline(np.mean(dv_mc), linestyle="--", label="MC mean")
    plt.axvline(mean_q, linestyle="--", label="GH mean")
    plt.xlabel(r"Total $\Delta v$ (AU/day)")
    plt.ylabel("Density")
    plt.title("Uncertainty propagation: Monte Carlo vs Gauss–Hermite quadrature")
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "uq_hist_mc_vs_quad.png"), dpi=300)
    plt.close()

    return float(np.mean(dv_mc)), float(np.var(dv_mc, ddof=1)), mean_q, var_q


def plot_uq_convergence(t0_best, tof_best, sig_t0, sig_tof):
    # Monte Carlo convergence (mean/var)
    mc_N = np.array([500, 1000, 2000, 5000, 10000, 20000])
    mc_mean = []
    mc_var = []

    for N in mc_N:
        xi = rng.standard_normal(int(N))
        eta = rng.standard_normal(int(N))
        dv = []
        for a, b in zip(xi, eta):
            t0 = t0_best + sig_t0 * a
            tof = tof_best + sig_tof * b
            try:
                val, _, _, _ = transfer_dv_and_miss(float(t0), float(tof), prograde=True)
                dv.append(val)
            except Exception:
                continue
        dv = np.array(dv)
        mc_mean.append(np.mean(dv))
        mc_var.append(np.var(dv, ddof=1))

    # Reference (take biggest MC as "truth")
    mean_ref = float(mc_mean[-1])
    var_ref = float(mc_var[-1])

    # GH tensor convergence: cost ~ n^2 model evals
    ns = np.array([3, 5, 7, 9, 11])
    q_cost = ns**2
    q_mean = []
    q_var = []
    for n in ns:
        m, v = uq_tensor_gh(t0_best, tof_best, sig_t0, sig_tof, int(n))
        q_mean.append(m)
        q_var.append(v)

    # Plot mean error
    plt.figure()
    plt.loglog(mc_N, np.abs(np.array(mc_mean) - mean_ref), marker="o", label="Monte Carlo")
    plt.loglog(q_cost, np.abs(np.array(q_mean) - mean_ref), marker="s", label="Tensor Gauss–Hermite")
    plt.xlabel("Model evaluations")
    plt.ylabel("Abs error in mean")
    plt.title("UQ convergence (mean): MC vs Gauss–Hermite quadrature")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "uq_convergence_mean.png"), dpi=300)
    plt.close()

    # Plot variance error
    plt.figure()
    plt.loglog(mc_N, np.abs(np.array(mc_var) - var_ref), marker="o", label="Monte Carlo")
    plt.loglog(q_cost, np.abs(np.array(q_var) - var_ref), marker="s", label="Tensor Gauss–Hermite")
    plt.xlabel("Model evaluations")
    plt.ylabel("Abs error in variance")
    plt.title("UQ convergence (variance): MC vs Gauss–Hermite quadrature")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "uq_convergence_var.png"), dpi=300)
    plt.close()


# =========================
# 8) MAIN
# =========================
def main():
    # --- Porkchop grid
    # Interpret t0 as "days from some epoch"
    t0_grid = np.linspace(0.0, 900.0, 180)      # 0..900 days
    tof_grid = np.linspace(120.0, 360.0, 160)   # 120..360 days

    print("Scanning porkchop grid (this may take ~1-3 minutes depending on your machine)...")
    DV, MISS = porkchop_scan(t0_grid, tof_grid)

    t0_best, tof_best, dv_best = find_best(DV, t0_grid, tof_grid)
    print("\nBest window found:")
    print(f"  t0_best  = {t0_best:.3f} days")
    print(f"  tof_best = {tof_best:.3f} days")
    print(f"  DV_best  = {dv_best:.6e} AU/day (scaled units)")

    # Plot porkchop + best point
    plot_porkchop(DV, t0_grid, tof_grid, best=(t0_best, tof_best, dv_best))

    # Plot best trajectory
    plot_best_trajectory(t0_best, tof_best)

    # Sensitivity plot (butterfly-style)
    plot_sensitivity_miss(t0_best, tof_best)

    # Uncertainty propagation around optimum
    # (These are small "navigation / scheduling" uncertainties)
    sig_t0 = 0.5   # days
    sig_tof = 1.0  # days

    mc_mean, mc_var, q_mean, q_var = plot_uq_mc_vs_quad(t0_best, tof_best, sig_t0, sig_tof)
    print("\nUncertainty propagation around optimum (DV as random variable):")
    print(f"  MC mean ≈ {mc_mean:.6e}, MC var ≈ {mc_var:.6e}")
    print(f"  GH mean ≈ {q_mean:.6e}, GH var ≈ {q_var:.6e}")

    plot_uq_convergence(t0_best, tof_best, sig_t0, sig_tof)

    print(f"\nSaved all figures to: {OUTDIR}/")
    print("  porkchop.png")
    print("  best_trajectory.png")
    print("  sensitivity_miss_distance.png")
    print("  uq_hist_mc_vs_quad.png")
    print("  uq_convergence_mean.png")
    print("  uq_convergence_var.png")


if __name__ == "__main__":
    main()
