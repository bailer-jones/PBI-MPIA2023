import numpy as np
import pylab as plt
from typing import Union
from scipy.stats import poisson


__author__ = "Morgan Fouesneau"
__copyright__ = "Copyright (c) 2022,  Max Planck Institute for Astronomy"
__credits__ = ["Morgan Fouesneau", "Coryn Bailer-Jones", "Ivelina Momcheva"]
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__maintainer__ = "Morgan Fouesneau"
__email__ = "fouesneau@mpia.de"
__status__ = "Production"


def signal(x: Union[float, np.array], 
           a: float, 
           b: float, 
           x0: float, 
           w: float, 
           t: float) -> Union[float, np.array]:
    """ Number of photons at position x 
    :param x: positions
    :param a: amplitude
    :param b: background
    :param x0: center of the signal
    :param w: Gaussian standard deviation of the signal
    :param t: proportional to the exposure time
    :returns: signal at position x given the parameters of the profile
    """
    return t * (a * np.exp( - 0.5 * (x - x0) ** 2 / w ** 2) + b)


def logupost(d: Union[float, np.array], 
             x: Union[float, np.array], 
             a: float, 
             b: float, 
             x0: float, 
             w: float, 
             t: float) -> Union[float, np.array]:
    """natural log posterior over (a,b).
    
    Prior on a and b: P(a,b) = const if a>0 and b>0, = 0 otherwise.
    Likelihood for one point is Poisson with mean d(x)
    Total likelihood is the product over x. 
    Unnormalized posterior is product of these.

    :param d: observed counts at positions x
    :param x: positions of same size as d
    :param a: amplitude
    :param b: background
    :param x0: center of the signal
    :param w: Gaussian standard deviation of the signal
    :param t: proportional to the exposure time
    """
    if ((a < 0) | (b < 0)):  # the effect of the prior
        return -float('inf')
    s = signal(x, a, b, x0, w, t)
    return sum(poisson.logpmf(d, s))



# Set model parameters (true and fixed)
x0 = 0     # center of peak
w = 1      # sd of peak
atrue = 2  # amplitude
btrue = 1  # background
t = 5      # scale factor (exposure time -> sets SNR)

# Simulate some data (by drawing from the likelihood)
np.random.seed(205)

xdat = np.arange(-7, 7, 0.5) * w
strue = signal(xdat, atrue, btrue, x0, w, t)
ddat  = np.random.poisson(strue)

# Define sampling grid to compute posterior (will be normalized over this range too). 
alim = 0.0, 4.0
blim = 0.5, 1.5
Nsamp = 100

a = np.linspace(alim[0], alim[1], Nsamp, endpoint=True)
b = np.linspace(blim[0], blim[1], Nsamp, endpoint=True)
z = np.array([[logupost(ddat, xdat, aj, bk, x0, w, t) for aj in a] for bk in b])

z -= z.max() # set maximum to zero

# Compute normalized marginalized posteriors, P(a|D) and P(b|D) 
# by summing over other parameter. Normalize by gridding.
delta_a = np.diff(a)[0]
delta_b = np.diff(b)[0]

p_a_D = np.sum(np.exp(z), 0)
p_a_D = p_a_D / (delta_a * p_a_D.sum())
p_b_D = np.sum(np.exp(z), 1)
p_b_D = p_b_D / (delta_b * p_b_D.sum())

# Compute mean, standard deviation, covariance, correlation, of a and b
mean_a = delta_a * np.sum(a * p_a_D)
mean_b = delta_b * np.sum(b * p_b_D)
sd_a = np.sqrt(delta_a * np.sum((a-mean_a) ** 2 * p_a_D))
sd_b = np.sqrt(delta_b * np.sum((b-mean_b) ** 2 * p_b_D))

# To calculate the covariance we need to normalize P(a,b|D) = exp(z).
# Here by brute force with two loops (there are better ways).
# The normalization constant is Z = delta_a * delta_b * sum(exp(z)).
# This is independent of (a, b) so can be calculated outside of the loops.
# The factor delta_a * delta_b will just cancel in the expression for 
# cov_ab, so we omit it entirely.

cov_ab = 0
for j, aj in enumerate(a):
  for k, bk in enumerate(b):
      cov_ab += (aj - mean_a) * (bk - mean_b) * np.exp(z[k,j])

cov_ab = cov_ab / np.sum(np.exp(z))
rho_ab = cov_ab / (sd_a * sd_b)
print(f"  a = {mean_a:0.3f} +/- {sd_a:0.3f}")
print(f"  b = {mean_b:0.3f} +/- {sd_b:0.3f}")
print(f"rho = {rho_ab:0.3f}")
#   a = 2.060 +/- 0.419
#   b = 0.988 +/- 0.100
# rho = -0.397

# Compute normalized conditional posteriors, P(a|b,D) and P(b|a,D)
# using true values of conditioned parameters. Vectorize(func, par)
# makes a vectorized function out of func in the parameter par.
p_a_bD = np.exp([logupost(ddat, xdat, aj, btrue, x0, w, t) for aj in a])
p_a_bD = p_a_bD / (delta_a * np.sum(p_a_bD))
p_b_aD = np.exp([logupost(ddat, xdat, atrue, bk, x0, w, t) for bk in b])
p_b_aD = p_b_aD / (delta_b * np.sum(p_b_aD))


# make figure
plt.figure(figsize=(6, 6))

plt.subplot(221)
xplot =  np.arange(-7, 7, 0.05) * w
plt.plot(xplot, signal(xplot, atrue, btrue, x0, w, t))
plt.xlabel("x")
plt.ylabel("s or d")
plt.plot(xdat, ddat, 'o', mfc="None", color='C0')

plt.subplot(222)
cs = plt.contour(np.exp(z), 6, extent=[min(a), max(a), min(b), max(b)], cmap='Blues')
plt.clabel(cs, cs.levels, inline=True, fontsize=10)
plt.hlines(btrue, min(a), max(a), color='0.5')
plt.vlines(atrue, min(b), max(b), color='0.5')
plt.xlabel('amplitude, a')
plt.ylabel('background, b')

# Plot the 1D marginalized posterior of b
plt.subplot(223)
plt.plot(b, p_b_D, ls='-')
plt.plot(b, p_b_aD, ls='--', color='C0')
plt.xlabel("background, b")
plt.ylabel("P(b | D)  and  P(b | a,D)")
plt.vlines(btrue, 0, max(p_b_aD), color='0.5')

# Plot the 1D marginalized posterior of a
plt.subplot(224)
plt.plot(a, p_a_D, ls='-')
plt.plot(a, p_a_bD, ls='--', color='C0')
plt.xlabel("amplitude, a")
plt.ylabel("P(a | D)  and  P(a | b,D)")
plt.vlines(atrue, 0, max(p_a_bD), color='0.5')

plt.tight_layout()
