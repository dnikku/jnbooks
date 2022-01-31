import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random, vmap, nn
from scipy.stats import gaussian_kde

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation

import pandas as pd

import pprint


numpyro.set_platform("cpu")

cm = 1/2.54 # centimeters in inches
px = 1/plt.rcParams['figure.dpi'] # pixel in inches


def standardize(array):
    return array / jnp.sum(array)

normalize = nn.normalize

def unif_prior(num):
    return standardize(jnp.repeat(1, num))

def grid_gen(num):
    return jnp.linspace(start=0, stop=1, num=num)


def HPDI(samples, prob):
    lo, hi = numpyro.diagnostics.hpdi(samples, prob=prob)
    print ("HDPI(%f) : [%f %f]" % (prob, lo, hi))
    return [lo, hi]
hpdi = numpyro.diagnostics.hpdi


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

def CDF(samples, interval):
    lo, hi = interval
    if lo != None and hi != None:
        prob = jnp.size(jnp.where((samples >= lo) & (samples <= hi))) / jnp.size(samples)
    elif lo != None:
        prob = jnp.size(jnp.where(samples >= lo)) / jnp.size(samples)
    elif hi != None:
        prob = jnp.size(jnp.where(samples <= hi)) / jnp.size(samples)
    else:
        raise Exception('(lo, hi) should not be both None.')
    prob_percent = int(prob * 10_000)/100
    print ("CDF(X in [%s %s]) : %f (%s%%)" % (hi, lo, prob, prob_percent))
    return prob


def PDF(X, values):
    return jnp.exp(X.log_prob(values))


def r_sample(X, num, **kwargs):
    num = (1,) if num is None else num
    rnd = random.PRNGKey(kwargs.get("r_seed")) if "r_seed" in kwargs else random_gen
    return X.sample(rnd, num)
    
def samples_grid(p_grid, posterior, num, **kwargs):
    return p_grid[r_sample(dist.Categorical(posterior), num, **kwargs)]


def dbinom(probs, N, K):
    return jnp.exp(dist.Binomial(total_count=N, probs=probs).log_prob(K))

def dbeta(p_grid, a, b):
    return jnp.exp(dist.Beta(a, b).log_prob(p_grid))

def rbinom(probs, N, num, **kwargs):
    return r_sample(dist.Binomial(total_count=N, probs=probs), num, **kwargs)

def runif(left, right, num, **kwargs):
    return r_sample(dist.Uniform(left, right), num, **kwargs)

def rnorm(mu, sigma, num, **kwargs):
    return r_sample(dist.Normal(mu, sigma), num, **kwargs)


random_gen = random.PRNGKey(100)

def random_reset(seed):
    global random_gen
    random_gen = random.PRNGKey(seed)
    
def rand(seed):
    return random.PRNGKey(seed)
    

## rendering

def fill_interval(X, Y, interval, title, **kwargs):
    low, hi = interval
    if "subplot" in kwargs:
        plt.subplot(kwargs.get("subplot"))
    plt.title(title)

    plt.plot(X, Y)
    plt.fill_between(X, Y, where=((X > low) & (X < hi)))
    
def pp(obj):
    pprint.pprint(obj, indent=2)
    
    
## rethinking
def precis(d):
    # tranform pandas
    d = dict(zip(d.columns, d.T.values)) if  isinstance(d, pd.core.frame.DataFrame) else d
    return numpyro.diagnostics.print_summary(d, 0.89, False)


def quap(flist, data, **kwargs):
    rnd = rand(kwargs.get("r_seed")) if "r_seed" in kwargs else random_gen
    steps = int(kwargs.get("steps", 2000))
    
    args = {}
    if "start" in kwargs:
        args["init_loc_fn"] = init_to_value(values=kwargs.get("start"))
    
    aprox =  AutoLaplaceApproximation(flist, **args)
    svi = SVI(flist, aprox, optim.Adam(1), Trace_ELBO(), **data)
    svi_result = svi.run(rnd, steps)
    return aprox, svi_result.params


                    