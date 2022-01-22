import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import gaussian_kde

import numpyro
import numpyro.distributions as dist


numpyro.set_platform("cpu")

cm = 1/2.54 # centimeters in inches
px = 1/plt.rcParams['figure.dpi'] # pixel in inches


def standardize(array):
    return array / jnp.sum(array)

def unif_prior(num):
    return standardize(jnp.repeat(1, num))

def grid_gen(num):
    return jnp.linspace(start=0, stop=1, num=num)


def HPDI(samples, prob):
    lo, hi = numpyro.diagnostics.hpdi(samples, prob=prob)
    print ("HDPI(%f) : [%f %f]" % (prob, lo, hi))
    return [lo, hi]

def PI(samples, prob_percent):
    lo, hi = jnp.percentile(samples, q=jnp.array([(100. - prob_percent)/2, (100. + prob_percent)/2]))
    print ("PI(%s) : [%f %f]" % (prob_percent, lo, hi))
    return [lo, hi]

def MAP(samples):
    # R: chainmode( samples,adj=0.01)
    m = samples[jnp.argmax(gaussian_kde(samples, bw_method=0.01)(samples))]
    print ("MAP : %f" % m)
    return m

def percentile(samples, percent):
    r = jnp.percentile(samples, jnp.array(percent) if isinstance(percent, list) else percent)
    print ("percentile(%%f) : %s" % (percent, r))
    return r

def sum_interval(samples, interval):
    lo, hi = interval
    if lo != None and hi != None:
        posterior = jnp.sum((samples > lo) & (samples < hi)) / len(samples)
    elif lo != None:
        posterior = jnp.sum(samples > lo) / len(samples)
    elif hi != None:
        posterior = jnp.sum(samples < hi) / len(samples)
    else:
        raise Exception('(lo, hi) should not be both None.')
    print ("CDF(%s) : %f" % (p, posterior))
    return posterior

def PDF(X, values):
    return jnp.exp(X.log_prob(values))

def r_sample(X, num):
    return X.sample(random_gen, num)

def dbinom(probs, N, K):
    return jnp.exp(dist.Binomial(total_count=N, probs=probs).log_prob(K))

def rbinom(probs, N, num):
    return dist.Binomial(total_count=N, probs=probs).sample(random_gen, num)

def dbeta(p_grid, a, b):
    return jnp.exp(dist.Beta(a, b).log_prob(p_grid))


random_gen = random.PRNGKey(100)

def random_reset(seed):
    global random_gen
    random_gen = random.PRNGKey(seed)

def samples_grid(p_grid, posterior, len):
    return p_grid[dist.Categorical(posterior).sample(random_gen, (10000,))]

## rendering

def fill_interval(X, Y, interval, title, **kwargs):
    low, hi = interval
    if "subplot" in kwargs:
        plt.subplot(kwargs.get("subplot"))
    plt.title(title)

    plt.plot(X, Y)
    plt.fill_between(X, Y, where=((X > low) & (X < hi)))