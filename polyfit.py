"""
Bayesian fitting of Polynomial model

%%file requirements.txt
numpy
scipy
matplotlib
emcee
arviz
"""
import arviz as az
import emcee
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyvander
from scipy.stats import norm

__author__ = "Morgan Fouesneau"
__copyright__ = "Copyright (c) 2022,  Max Planck Institute for Astronomy"
__credits__ = ["Morgan Fouesneau", "Coryn Bailer-Jones", "Ivelina Momcheva"]
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__maintainer__ = "Morgan Fouesneau"
__email__ = "fouesneau@mpia.de"
__status__ = "Production"



def polynomial_model(
    deg: int,
    coeffs: np.array,
    x: np.array
):
    """
    Generate a polynomial model y = sum(a[k] * x ** k, k=[0..deg])

    :param deg: degree of the polynomial
    :param coeffs: coefficients of the polynomials
    :param x: evaluation points or the polynomial expanded version.
    :returns: polynomial model evaluated at x
    """
    if len(coeffs) != (deg + 1):
        raise AttributeError(
            f"The coefficient vector is not of length 'deg':{deg + 1} vs. {coeffs.shape}")
    if np.ndim(x) == 1:
        X = polyvander(x, deg)  # [1, x, x**2, ..., x ** deg]
        return X @ np.array(coeffs)
    return x @ np.array(coeffs)


def lnprior(
        deg: int,
        coeffs: np.array,
        l1: float = 0.,
        l2: float = 0.) -> float:
    """ln Prior on the polynomial parameters: ln p(coeffs)

    Generate a polynomial model y = sum(a[k] * x ** k, k=[0..deg])

    :param deg: degree of the polynomial
    :param coeffs: coefficients of the polynomials
    :param l1: l1 norm contribution
    :param l2: l2 norm contribution
    :return: ln-prior value
    """
    if (l1 == 0) and (l2 == 0):
        return 0.
    return np.log(np.sum(l1 * np.abs(coeffs) + 0.5 * l2 * coeffs ** 2))


def lnlikelihood(
        coeffs: np.array,
        x: np.array,
        y: np.array,
        sy: np.array,
        deg: int):
    """ ln Likelihood:  ln p(y | x, sy, coeffs)

    :param coeffs: coefficients of the polynomials
    :param x: evaluation points
    :param y: observed values
    :param sy: observed value uncertainties
    :param deg: degree of the polynomial
    :return: ln-likelihood
    """
    ypred = polynomial_model(deg, coeffs, x)
    return np.sum(norm.logpdf(y, loc=ypred, scale=sy))


def lnposterior(
        coeffs: np.array,
        x: np.array,
        y: np.array,
        sy: np.array,
        deg: int):
    """ ln posterior:   ln p(y | x, sy, coeffs) + ln p(coeffs)

    :param coeffs: coefficients of the polynomials
    :param x: evaluation points
    :param y: observed values
    :param sy: observed value uncertainties
    :param deg: degree of the polynomial
    :return: ln-likelihood
    """
    return lnprior(deg, coeffs) + lnlikelihood(coeffs, x, y, sy, deg)


# mock data
np.random.seed(123)
N = 10
x = np.sort(np.random.uniform(-5, 5, N))
ctrue = np.array([0, 1, -2, 0.5])
deg = len(ctrue) - 1
sy = 5 * np.random.rand(N)
ytrue = polynomial_model(deg, ctrue, x)
y = ytrue + np.random.normal(0, sy)

# plot the mock data
xplot = np.linspace(-5, 5, 1000)
yplot = polynomial_model(deg, ctrue, xplot)
plt.plot(xplot, yplot, color='0.5', label='ytrue')
plt.errorbar(x, y, yerr=sy, linestyle='None',
             marker='o', mfc='w', label='yobs')
plt.ylim(y.min() - 5, y.max() + 5)
plt.legend(loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('y')

# Proceed to mcmc fitting

ndim, nwalkers = deg + 1, 10
p0 = np.random.randn(nwalkers, ndim)

X = polyvander(x, deg)  # precompute
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, lnposterior, args=[X, y, sy, deg])


# We could run sampler.run_mcmc(p0, 2000) but sometimes it could be slow,
# and it is nice to monitor what happens

# Let's run the burn-in part
for state in sampler.sample(p0, iterations=1_000, progress=True):
    pass
sampler.reset()  # tells the sampler to forget the chains

# Restart from the current state of the chains
for _ in sampler.sample(state, iterations=1_000, progress=True):
    pass


# Explore the outputs

names = ["1", "x"] + [rf"$x^{k}$" for k in range(2, deg+1)]
data = az.from_emcee(sampler, var_names=names, )

axes = az.plot_pair(
    data,
    # var_names=names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    kde_kwargs={
        # "hdi_probs": [0.682, 0.954, 0.997],  # Plot HDI contours
        "hdi_probs": [0.682, 0.954, 0.997],  # Plot HDI contours
        "contourf_kwargs": {"cmap": "Blues"},
    },
    figsize=(8, 6)
)

# add true values for comparison
for dim in range(len(ctrue)):
    val = ctrue[dim]
    for ax in axes[:, dim]:
        lim = ax.get_ylim()
        ax.vlines([val], *lim, color='r')
        ax.set_ylim(lim)

for dim in range(1, len(ctrue)):
    val = ctrue[dim]
    for ax in axes[dim, :dim]:
        lim = ax.get_xlim()
        ax.hlines([val], *lim, color='r')
        ax.set_xlim(lim)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

# plot the ppc
xplot = np.linspace(-5, 5, 1000)
yplot = polynomial_model(deg, ctrue, xplot)

params = sampler.flatchain[-1000:]
Xpred = polyvander(xplot, deg)
ypred = np.array([polynomial_model(deg, pos, Xpred) for pos in params])

plt.plot(xplot, ypred.T, rasterized=True, color='k', alpha=0.2, lw=0.1)
plt.plot(xplot, yplot, color='w', lw=5)
plt.plot(xplot, yplot, color='C0', lw=3, label='ytrue')

plt.errorbar(x, y, yerr=sy, linestyle='None',
             marker='o', mfc='w', label='yobs')
plt.ylim(y.min() - 5, y.max() + 5)
plt.legend(loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('y')
