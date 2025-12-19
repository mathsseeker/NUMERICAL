import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss   # Gauss–Legendre
from numpy.polynomial.hermite import hermgauss   # Gauss–Hermite
import os


# =====================================
# 1. BASIC QUADRATURE UTILITIES
# =====================================

def trap_rule(f, a, b, n):
    """
    Composite trapezoidal rule on [a,b] with n subintervals.
    Returns approximation and number of function evaluations.
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    I = h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])
    n_eval = len(x)
    return I, n_eval


def simpson_rule(f, a, b, n):
    """
    Composite Simpson's rule on [a,b] with n subintervals (n must be even).
    Returns approximation and number of function evaluations.
    """
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires n to be even.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    I = (h / 3.0) * (y[0] + y[-1] +
                     4.0 * y[1:-1:2].sum() +
                     2.0 * y[2:-1:2].sum())
    n_eval = len(x)
    return I, n_eval


# =====================================
# 2. GAUSS–LEGENDRE QUADRATURE
# =====================================

def gauss_legendre(f, n, a=-1.0, b=-1.0):
    """
    n-point Gauss–Legendre quadrature for ∫_a^b f(x) dx.
    Returns approximation and number of function evaluations (= n).
    """
    if a == -1.0 and b == -1.0:
        # default to [-1,1] if not specified
        a, b = -1.0, 1.0

    # nodes and weights on [-1,1]
    xi, wi = leggauss(n)
    # map to [a,b]
    xm = 0.5 * (b - a) * xi + 0.5 * (a + b)
    wm = 0.5 * (b - a) * wi
    I = np.sum(wm * f(xm))
    n_eval = n
    return I, n_eval


# =====================================
# 3. GAUSS–CHEBYSHEV QUADRATURE
# =====================================

def chebyshev_nodes_weights(n):
    """
    n-point Gauss–Chebyshev (first kind) nodes & weights on [-1,1]
    for integrals of the form ∫_{-1}^1 f(x) / sqrt(1 - x^2) dx.
    Nodes: x_k = cos((2k-1)π/(2n)), weights: w_k = π/n.
    """
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * np.pi / (2 * n))
    w = np.full(n, np.pi / n)
    return x, w


def gauss_chebyshev(f, n):
    """
    n-point Gauss–Chebyshev quadrature for
    I = ∫_{-1}^1 f(x) / sqrt(1 - x^2) dx.
    Returns approximation and function evals (= n).
    """
    x, w = chebyshev_nodes_weights(n)
    I = np.sum(w * f(x))
    n_eval = n
    return I, n_eval


def chebyshev_trap(f, n):
    """
    Trapezoidal rule for Chebyshev-type integral
    using change of variables x = cos(theta), theta ∈ [0,π]:
        ∫_{-1}^1 f(x)/sqrt(1-x^2) dx = ∫_0^π f(cos θ) dθ
    """
    def g(theta):
        return f(np.cos(theta))
    I, n_eval = trap_rule(g, 0.0, np.pi, n)
    return I, n_eval


def chebyshev_simpson(f, n):
    """
    Simpson rule in theta-space for Chebyshev-type integral
    (same substitution as chebyshev_trap).
    """
    def g(theta):
        return f(np.cos(theta))
    I, n_eval = simpson_rule(g, 0.0, np.pi, n)
    return I, n_eval


# =====================================
# 4. GAUSS–HERMITE QUADRATURE
# =====================================

def gauss_hermite(f, n):
    """
    n-point Gauss–Hermite quadrature for
    I = ∫_{-∞}^∞ f(x) e^{-x^2} dx.
    Returns approximation and number of function evals (= n).
    """
    xi, wi = hermgauss(n)  # nodes and weights for weight e^{-x^2}
    I = np.sum(wi * f(xi))
    n_eval = n
    return I, n_eval


def hermite_trap(f, n, L=5.0):
    """
    Trapezoidal rule on [-L, L] for Hermite-type integral:
        ∫_{-∞}^∞ f(x)e^{-x^2}dx ≈ ∫_{-L}^L f(x)e^{-x^2}dx
    """
    def g(x):
        return f(x) * np.exp(-x**2)
    I, n_eval = trap_rule(g, -L, L, n)
    return I, n_eval


def hermite_simpson(f, n, L=5.0):
    """
    Simpson rule on [-L, L] for Hermite-type integral
    (same truncated integral as hermite_trap).
    """
    def g(x):
        return f(x) * np.exp(-x**2)
    I, n_eval = simpson_rule(g, -L, L, n)
    return I, n_eval


# =====================================
# 5. TEST FUNCTIONS
# =====================================

def f_exp(x):
    return np.exp(x)

def f_sin(x):
    return np.sin(x)

def f_x4(x):
    return x**4


# =====================================
# 6. REFERENCE INTEGRALS (HIGH-ORDER GAUSS)
# =====================================

def reference_legendre(f):
    """
    Reference for ∫_{-1}^1 f(x) dx via high-order Gauss–Legendre.
    """
    I_ref, _ = gauss_legendre(f, 200, -1.0, 1.0)
    return I_ref

def reference_chebyshev(f):
    """
    Reference for ∫_{-1}^1 f(x)/sqrt(1-x^2) dx via high-order Gauss–Chebyshev.
    """
    x, w = chebyshev_nodes_weights(200)
    return np.sum(w * f(x))

def reference_hermite(f):
    """
    Reference for ∫_{-∞}^∞ f(x)e^{-x^2} dx via high-order Gauss–Hermite.
    """
    I_ref, _ = gauss_hermite(f, 200)
    return I_ref


# =====================================
# 7. ERROR STUDY HELPERS
# =====================================

def safe_errors(errors, eps=1e-18):
    """Replace zero/negative errors by eps so loglog plots work."""
    return [max(abs(e), eps) for e in errors]


def error_study_legendre(f, name="f"):
    """
    Compare Gauss–Legendre vs Trapezoidal vs Simpson
    for ∫_{-1}^1 f(x) dx.
    Saves a log-log plot of error vs function evaluations.
    """
    exact = reference_legendre(f)

    N_list = [2, 4, 8, 16, 32]

    err_gauss, eval_gauss = [], []
    err_trap,  eval_trap  = [], []
    err_simp,  eval_simp  = [], []

    a, b = -1.0, 1.0

    for n in N_list:
        # Gauss–Legendre
        I_g, ne_g = gauss_legendre(f, n, a, b)
        err_gauss.append(I_g - exact)
        eval_gauss.append(ne_g)

        # Trapezoidal
        I_t, ne_t = trap_rule(f, a, b, n)
        err_trap.append(I_t - exact)
        eval_trap.append(ne_t)

        # Simpson
        I_s, ne_s = simpson_rule(f, a, b, n)
        err_simp.append(I_s - exact)
        eval_simp.append(ne_s)

    plt.figure()
    plt.loglog(eval_gauss, safe_errors(err_gauss), 'o-', label="Gauss–Legendre")
    plt.loglog(eval_trap,  safe_errors(err_trap),  's--', label="Trapezoidal")
    plt.loglog(eval_simp,  safe_errors(err_simp),  'd-.', label="Simpson")
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Absolute error")
    plt.title(f"Legendre-type integral: ∫_(-1)^1 {name}(x) dx")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.savefig(f"CHARTS/legendre_{name.replace('^','').replace('(','').replace(')','')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def error_study_chebyshev(f, name="f"):
    """
    Compare Gauss–Chebyshev vs Trapezoidal vs Simpson
    for ∫_{-1}^1 f(x)/sqrt(1-x^2) dx.
    Trapezoid & Simpson are implemented in θ-space (x = cos θ).
    """
    exact = reference_chebyshev(f)

    N_list = [2, 4, 8, 16, 32]

    err_gauss, eval_gauss = [], []
    err_trap,  eval_trap  = [], []
    err_simp,  eval_simp  = [], []

    for n in N_list:
        # Gauss–Chebyshev
        I_g, ne_g = gauss_chebyshev(f, n)
        err_gauss.append(I_g - exact)
        eval_gauss.append(ne_g)

        # Trapezoid in θ-space
        I_t, ne_t = chebyshev_trap(f, n)
        err_trap.append(I_t - exact)
        eval_trap.append(ne_t)

        # Simpson in θ-space
        I_s, ne_s = chebyshev_simpson(f, n)
        err_simp.append(I_s - exact)
        eval_simp.append(ne_s)

    plt.figure()
    plt.loglog(eval_gauss, safe_errors(err_gauss), 'o-', label="Gauss–Chebyshev")
    plt.loglog(eval_trap,  safe_errors(err_trap),  's--', label="Trapezoidal (θ-space)")
    plt.loglog(eval_simp,  safe_errors(err_simp),  'd-.', label="Simpson (θ-space)")
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Absolute error")
    plt.title(r"Chebyshev-type integral: $\int_{-1}^1 \frac{%s(x)}{\sqrt{1-x^2}} dx$" % name)
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.savefig(f"CHARTS/chebyshev_{name.replace('^','').replace('(','').replace(')','')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def error_study_hermite(f, name="f"):
    """
    Compare Gauss–Hermite vs Trapezoidal vs Simpson
    for ∫_{-∞}^∞ f(x)e^{-x^2} dx.
    Trapezoid/Simpson are done on [-L,L] (truncated domain).
    """
    exact = reference_hermite(f)

    N_list = [2, 4, 8, 16, 32]
    L = 5.0

    err_gauss, eval_gauss = [], []
    err_trap,  eval_trap  = [], []
    err_simp,  eval_simp  = [], []

    for n in N_list:
        # Gauss–Hermite (no truncation)
        I_g, ne_g = gauss_hermite(f, n)
        err_gauss.append(I_g - exact)
        eval_gauss.append(ne_g)

        # Trapezoid on [-L,L]
        I_t, ne_t = hermite_trap(f, n, L=L)
        err_trap.append(I_t - exact)
        eval_trap.append(ne_t)

        # Simpson on [-L,L]
        I_s, ne_s = hermite_simpson(f, n, L=L)
        err_simp.append(I_s - exact)
        eval_simp.append(ne_s)

    plt.figure()
    plt.loglog(eval_gauss, safe_errors(err_gauss), 'o-', label="Gauss–Hermite")
    plt.loglog(eval_trap,  safe_errors(err_trap),  's--', label="Trapezoidal (truncated)")
    plt.loglog(eval_simp,  safe_errors(err_simp),  'd-.', label="Simpson (truncated)")
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Absolute error")
    plt.title(r"Hermite-type integral: $\int_{-\infty}^{\infty} %s(x) e^{-x^2} dx$" % name)
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.savefig(f"CHARTS/hermite_{name.replace('^','').replace('(','').replace(')','')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


# =====================================
# 8. MAIN: RUN ALL EXPERIMENTS
# =====================================

if __name__ == "__main__":
    os.makedirs("CHARTS", exist_ok=True)

    tests = [
        (f_exp, "e^x"),
        (f_sin, "sin(x)"),
        (f_x4, "x^4")
    ]

    # Legendre tests: ∫_{-1}^1 f(x) dx
    for f, name in tests:
        print(f"[Legendre]  {name}")
        error_study_legendre(f, name=name)

    # Chebyshev tests: ∫_{-1}^1 f(x)/sqrt(1-x^2) dx
    for f, name in tests:
        print(f"[Chebyshev] {name}")
        error_study_chebyshev(f, name=name)

    # Hermite tests: ∫_{-∞}^∞ f(x) e^{-x^2} dx
    for f, name in tests:
        print(f"[Hermite]   {name}")
        error_study_hermite(f, name=name)
