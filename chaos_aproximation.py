import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import factorial, sqrt, pi
import matplotlib.pyplot as plt

# =========================================================
# 1. GAUSS–HERMITE EXPECTATION FOR STANDARD NORMAL
# =========================================================

def gh_expectation_standard_normal(func, n_quad=50):
    """
    Approximate E[f(X)] where X ~ N(0,1) using n_quad-point
    Gauss–Hermite quadrature.

    We use the relation:
        E[g(X)] = 1/sqrt(pi) ∫ g(sqrt(2)*y) e^{-y^2} dy

    hermgauss returns nodes/weights for ∫ f(y)e^{-y^2}dy.
    """
    # Nodes y, weights w for weight e^{-y^2}
    y, w = hermgauss(n_quad)
    x = np.sqrt(2.0) * y          # map to standard normal
    w_sn = w / np.sqrt(pi)        # adjusted weights

    return np.sum(w_sn * func(x))


def gh_nodes_weights_standard_normal(n_quad=50):
    """
    Return nodes x_i and weights w_i such that
    E[g(X)] ≈ sum_i w_i g(x_i) for X ~ N(0,1),
    using n_quad-point Gauss–Hermite quadrature.
    """
    y, w = hermgauss(n_quad)
    x = np.sqrt(2.0) * y
    w_sn = w / np.sqrt(pi)
    return x, w_sn


# =========================================================
# 2. PROBABILISTS' HERMITE POLYNOMIALS & ORTHONORMAL BASIS
# =========================================================

def hermite_probabilists(n, x):
    """
    Evaluate the n-th probabilists' Hermite polynomial He_n(x)
    at points x (array or scalar) using recurrence:

        He_0(x) = 1
        He_1(x) = x
        He_{k+1}(x) = x He_k(x) - k He_{k-1}(x)

    Returns an array with same shape as x.
    """
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x

    He_nm1 = np.ones_like(x)  # He_0
    He_n = x.copy()           # He_1

    for k in range(1, n):
        He_np1 = x * He_n - k * He_nm1
        He_nm1, He_n = He_n, He_np1

    return He_n


def psi_orthonormal(n, x):
    """
    Orthonormal Hermite basis function for X~N(0,1):
        psi_n(x) = He_n(x) / sqrt(n!)

    where He_n are probabilists' Hermite polynomials.
    """
    he = hermite_probabilists(n, x)
    return he / sqrt(factorial(n))


# =========================================================
# 3. MODEL: THE "CHAOS" EXAMPLE
# =========================================================

def model_y(x):
    """
    Nonlinear model Y = sin(X) + 0.3 X^2
    with X ~ N(0,1).
    """
    return np.sin(x) + 0.3 * x**2


# =========================================================
# 4. BUILD POLYNOMIAL CHAOS EXPANSION
# =========================================================

def build_pce(max_order=5, n_quad=50):
    """
    Build a 1D Polynomial Chaos Expansion for Y = model_y(X),
    with X ~ N(0,1), using orthonormal Hermite basis up to
    order max_order.

    Returns
    -------
    coeffs : ndarray, shape (max_order+1,)
        PCE coefficients c_k.
    """
    # Quadrature nodes and weights for X~N(0,1)
    xq, wq = gh_nodes_weights_standard_normal(n_quad)

    yq = model_y(xq)

    coeffs = np.zeros(max_order + 1)

    for k in range(max_order + 1):
        psi_k = psi_orthonormal(k, xq)
        # c_k = E[Y psi_k(X)] ≈ sum w_i y_i psi_k_i
        coeffs[k] = np.sum(wq * yq * psi_k)

    return coeffs


def pce_evaluate(coeffs, x):
    """
    Evaluate the PCE surrogate at points x.

    Y_hat(x) = sum_{k=0}^p c_k psi_k(x)
    """
    x = np.asarray(x)
    p = len(coeffs) - 1
    y_hat = np.zeros_like(x, dtype=float)
    for k in range(p + 1):
        y_hat += coeffs[k] * psi_orthonormal(k, x)
    return y_hat


def pce_mean_var(coeffs):
    """
    For an orthonormal basis, the mean and variance of the PCE are:

        mean    = c_0
        variance = sum_{k>=1} c_k^2
    """
    mean = coeffs[0]
    var = np.sum(coeffs[1:]**2)
    return mean, var


# =========================================================
# 5. MONTE CARLO REFERENCE
# =========================================================

def monte_carlo_stats(n_samples=100_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_samples)
    Y = model_y(X)
    mean = Y.mean()
    var = Y.var()
    return mean, var, X, Y


# =========================================================
# 6. PLOTTING FUNCTIONS
# =========================================================

def plot_true_vs_pce(coeffs):
    x = np.linspace(-3, 3, 400)
    y_true = model_y(x)
    y_pce = pce_evaluate(coeffs, x)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y_true, label="True model Y(x)")
    plt.plot(x, y_pce, '--', label="PCE surrogate", linewidth=2)
    plt.title("True Model vs Polynomial Chaos Surrogate")
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histograms(coeffs, X_mc, Y_mc):
    Y_pce = pce_evaluate(coeffs, X_mc)

    plt.figure(figsize=(7, 4))
    plt.hist(Y_mc, bins=60, alpha=0.5, density=True, label="Monte Carlo Y")
    plt.hist(Y_pce, bins=60, alpha=0.5, density=True, label="PCE surrogate Y")
    plt.title("Distribution of Y: Monte Carlo vs PCE")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_coefficients(coeffs):
    plt.figure(figsize=(6, 4))
    plt.stem(range(len(coeffs)), coeffs)  # removed use_line_collection
    plt.title("PCE Coefficients c_k")
    plt.xlabel("Order k")
    plt.ylabel("Coefficient c_k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_mean_variance(mean_pce, var_pce, mean_mc, var_mc):
    labels = ["Mean", "Variance"]
    mc_vals = [mean_mc, var_mc]
    pce_vals = [mean_pce, var_pce]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, mc_vals, width, label="Monte Carlo")
    plt.bar(x + width/2, pce_vals, width, label="PCE")
    plt.xticks(x, labels)
    plt.title("PCE vs MC: First Two Moments")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_pce_convergence(max_order=10, n_quad=50, n_mc=200_000):
    rng = np.random.default_rng(0)
    X = rng.standard_normal(n_mc)
    Y = model_y(X)

    errors = []

    for p in range(max_order + 1):
        coeffs = build_pce(p, n_quad=n_quad)
        Y_hat = pce_evaluate(coeffs, X)
        error = np.sqrt(np.mean((Y - Y_hat)**2))
        errors.append(error)

    plt.figure(figsize=(7, 4))
    plt.semilogy(range(max_order + 1), errors, '-o')
    plt.title("Convergence of PCE with Polynomial Order")
    plt.xlabel("Polynomial Order p")
    plt.ylabel("L2 Error (Monte Carlo estimate)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 7. DEMO / MAIN
# =========================================================

def demo_pce():
    # Parameters
    max_order = 5     # PCE order
    n_quad = 60       # GH quadrature order
    n_mc = 100_000    # Monte Carlo samples

    print(f"Building PCE of order p={max_order} using n_quad={n_quad} Gauss–Hermite points...")
    coeffs = build_pce(max_order=max_order, n_quad=n_quad)
    mean_pce, var_pce = pce_mean_var(coeffs)

    print("\nPCE coefficients c_k:")
    for k, ck in enumerate(coeffs):
        print(f"  c_{k} = {ck:.6e}")

    print("\nPCE mean and variance (from coefficients):")
    print(f"  mean_PCE = {mean_pce:.6e}")
    print(f"  var_PCE  = {var_pce:.6e}")

    # Monte Carlo reference
    print(f"\nRunning Monte Carlo with n={n_mc} samples for reference...")
    mean_mc, var_mc, X_mc, Y_mc = monte_carlo_stats(n_samples=n_mc)

    print("Monte Carlo mean and variance:")
    print(f"  mean_MC  = {mean_mc:.6e}")
    print(f"  var_MC   = {var_mc:.6e}")

    # Compare
    print("\nRelative errors (PCE vs MC):")
    rel_err_mean = abs(mean_pce - mean_mc) / abs(mean_mc)
    rel_err_var = abs(var_pce - var_mc) / abs(var_mc)
    print(f"  rel_error_mean = {rel_err_mean:.3e}")
    print(f"  rel_error_var  = {rel_err_var:.3e}")

    # L2 error of surrogate over distribution (estimated via MC)
    Y_hat_mc = pce_evaluate(coeffs, X_mc)
    l2_error = np.sqrt(np.mean((Y_mc - Y_hat_mc)**2))
    print(f"\nEstimated L2 error between Y and PCE surrogate (MC): {l2_error:.3e}")

    # ==== Plots ====
    print("\nGenerating plots...")
    plot_true_vs_pce(coeffs)
    plot_histograms(coeffs, X_mc, Y_mc)
    plot_coefficients(coeffs)
    plot_mean_variance(mean_pce, var_pce, mean_mc, var_mc)
    plot_pce_convergence()


if __name__ == "__main__":
    demo_pce()
