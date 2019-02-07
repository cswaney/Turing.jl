---
title: Sampler Visualization
permalink: /docs/sampler-viz/
toc: true
toc_sticky: true
mathjax: true
---

<a id='Introduction-1'></a>

## Introduction


<a id='The-Code-1'></a>

## The Code


For each sampler, we will use the same code to plot sampler paths. The block below loads the relevant libraries and defines a function for plotting the sampler's trajectory across the posterior.


The Turing model definition used here is not especially practical, but it is designed in such a way as to produce visually interesting posterior surfaces to show how different samplers move along the distribution.


```julia
using Plots
using StatsPlots
using Turing
using Bijectors
using Random

Random.seed!(0)

# Define a strange model.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    bumps = sin(m) + cos(m)
    m = m + 5*bumps
    for i in eachindex(x)
      x[i] ~ Normal(m, sqrt(s))
    end
    return s, m
end

# Define our data points.
x = [1.5, 2.0, 13.0, 2.1, 0.0]

# Set up the model call, sample from the prior.
model = gdemo(x)
vi = Turing.VarInfo()
model(vi, Turing.SampleFromPrior())
vi.flags["trans"] = [true, false]

# Evaluate surface at coordinates.
function evaluate(m1, m2)
    vi.vals .= [m1, m2]
    model(vi, Turing.SampleFromPrior())
    -vi.logp
end

function plot_sampler(chain)
    # Extract values from chain.
    ss = link.(Ref(InverseGamma(2,3)), chain[:s])
    ms = chain[:m]
    lps = chain[:lp]

    # How many surface points to sample.
    granularity = 500

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss); σ_stop = maximum(ss) + spread * std(ss);
    μ_start = minimum(ms) - spread * std(ms); μ_stop = maximum(ms) + spread * std(ms);
    σ_rng = collect(range(σ_start, stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start, stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(σ_rng, μ_rng, evaluate,
          camera=(30, 65),
          ticks=nothing,
          colorbar=false,
          color=:inferno)

    line_range = 1:length(ms)

    plot3d!(ss[line_range], ms[line_range], -lps[line_range],
        lc =:viridis, line_z=collect(line_range),
        legend=false, colorbar=false, alpha=0.5)

    return p
end
```


<a id='Samplers-1'></a>

## Samplers


<a id='Gibbs-1'></a>

### Gibbs


Gibbs sampling tends to exhibit a "jittery" trajectory. The example below combines `HMC` and `PG` sampling to traverse the posterior.


```julia
c = sample(model, Gibbs(1000,
  HMC(1, 0.01, 5, :s), PG(20, 1, :m)))
plot_sampler(c)
```


![](/assets/figures/samplers_2_1.svg)


Other samplers can be combined as well:


```julia
c = sample(model, Gibbs(1000, MH(1, :s), SGLD(100, 0.01, :m)))
plot_sampler(c)
```


![](/assets/figures/samplers_3_1.svg)


<a id='HMC-1'></a>

### HMC


Hamiltonian Monte Carlo (HMC) sampling is a typical sampler to use, as it tends to be fairly good at converging in a efficient manner. It can often be tricky to set the correct parameters for this sampler however, and the `NUTS` sampler is often easier to run if you don't want to spend too much time fiddling with step size and and the number of steps to take.


```julia
c = sample(model, HMC(1000, 0.01, 10))
```


```
[HMC] Finished with
  Running time        = 2.4334476359999986;
  Accept rate         = 1.0;
  #lf / sample        = 9.99;
  #evals / sample     = 11.99;
  pre-cond. metric    = [1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_4_1.svg)


<a id='HMCDA-1'></a>

### HMCDA


The HMCDA sampler is an implementation of the Hamiltonian Monte Carlo with Dual Averaging algorithm found in the paper "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" by Hoffman and Gelman (2011). The paper can be found on [arXiv](https://arxiv.org/abs/1111.4246) for the interested reader.


```julia
c = sample(model, HMCDA(1000, 200, 0.65, 0.3))
```


```
[HMCDA] Finished with
  Running time        = 2.0255241690000028;
  Accept rate         = 0.679;
  #lf / sample        = 1.056;
  #evals / sample     = 3.062;
  pre-cond. metric    = [1.0, 1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_5_1.svg)


<a id='MH-1'></a>

### MH


Metropolis-Hastings (MH) sampling is one of the earliest Markov Chain Monte Carlo methods. MH sampling does not "move" a lot, unlike many of the other samplers implemented in Turing. Typically a much longer chain is required to converge to an appropriate parameter estimate.


The plot below only uses 1,000 iterations of Metropolis-Hastings.


```julia
c = sample(model, MH(1000))
```


```
[MH] Finished with
  Running time        = 0.032115146000000046;
  Accept rate         = 0.015;
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_6_1.svg)


As you can see, the MH sampler doesn't move parameter estimates very often.


<a id='NUTS-1'></a>

### NUTS


The No U-Turn Sampler (NUTS) is an implementation of the algorithm found in the paper "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" by Hoffman and Gelman (2011). The paper can be found on [arXiv](https://arxiv.org/abs/1111.4246) for the interested reader.


NUTS tends to be very good at traversing the minima of complex posteriors quickly.


```julia
c = sample(model, NUTS(1000, 0.65))
```


```
[NUTS] Finished with
  Running time        = 2.7755298859999935;
  #lf / sample        = 0.002;
  #evals / sample     = 12.313;
  pre-cond. metric    = [1.0, 1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_7_1.svg)


The only parameter that needs to be set other than the number of iterations to run is the target acceptance rate. In the Hoffman and Gelman paper, they note that a target acceptance rate of 0.65 is typical.


Here is a plot showing a very high acceptance rate. Note that it appears to "stick" to a locla minima and is not particularly good at exploring the posterior.


```julia
c = sample(model, NUTS(1000, 0.95))
```


```
[NUTS] Finished with
  Running time        = 1.7990740579999978;
  #lf / sample        = 0.004;
  #evals / sample     = 23.805;
  pre-cond. metric    = [1.0, 1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_8_1.svg)


An exceptionally low acceptance rate will show very few moves on the posterior:


```julia
c = sample(model, NUTS(1000, 0.2))
```


```
[NUTS] Finished with
  Running time        = 0.5854548800000005;
  #lf / sample        = 0.002;
  #evals / sample     = 6.627;
  pre-cond. metric    = [1.0, 1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_9_1.svg)


<a id='PG-1'></a>

### PG


The Particle Gibbs (PG) sampler is an implementation of an algorithm from the paper "Particle Markov chain Monte Carlo methods" by Andrieu, Doucet, and Holenstein (2010). The interested reader can learn more [here](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2009.00736.x).


The two parameters are the number of particles, and the number of iterations. The plot below shows the use of 20 particles.


```julia
c = sample(model, PG(20, 1000))
plot_sampler(c)
```


![](/assets/figures/samplers_10_1.svg)


Next, we plot using 50 particles.


```julia
c = sample(model, PG(50, 1000))
plot_sampler(c)
```


![](/assets/figures/samplers_11_1.svg)


<a id='PMMH-1'></a>

### PMMH


The Particle Marginal Metropolis-Hastings (PMMH) sampler is an implementation of an algorithm from the paper "Particle Markov chain Monte Carlo methods" by Andrieu, Doucet, and Holenstein (2010). The interested reader can learn more [here](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2009.00736.x).


PMMH supports the use of different samplers across different parameter spaces, similar to the Gibbs sampler. The plot below uses SMC and MH.


```julia
c = sample(model, PMMH(1000, SMC(20, :m), MH(10,:s)))
```


```
[PMMH] Finished with
  Running time    = 4.950841641000002;
  Accept rate         = 0.09;
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_12_1.svg)


<a id='PIMH-1'></a>

### PIMH


In addition to PMMH, Turing also support the Particle Independent Metropolis-Hastings (PIMH). PIMH accepts a number of iterations, and an SMC call.


```julia
c = sample(model, PIMH(1000, SMC(20)))
```


```
[PMMH] Finished with
  Running time    = 4.422653851000006;
  Accept rate         = 0.232;
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_13_1.svg)


<a id='SGHMC-1'></a>

### SGHMC


Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) tends to produce sampling paths not unlike that of stochastic gradient descent in other machine learning model types. It is an implementation of an algorithm in the paper "Stochastic Gradient Hamiltonian Monte Carlo" by Chen, Fox, and Guestrin (2014). The interested reader can learn more [here](https://arxiv.org/abs/1402.4102). This sampler is very similar to the SGLD sampler below.


The two parameters used in SGHMC are the learing rate and the momentum decay. Here is sampler with a higher momentum decay of 0.1:


```julia
c = sample(model, SGHMC(1000, 0.001, 0.1))
```


```
[SGHMC] Finished with
  Running time        = 1.2120846790000028;
  Accept rate         = 1.0;
  #lf / sample        = 0.0;
  #evals / sample     = 501.5;
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_14_1.svg)


And the same sampler with a much lower momentum decay:


```julia
c = sample(model, SGHMC(1000, 0.001, 0.01))
```


```
[SGHMC] Finished with
  Running time        = 0.08846964400000006;
  Accept rate         = 1.0;
  #lf / sample        = 0.0;
  #evals / sample     = 501.5;
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_15_1.svg)


<a id='SGLD-1'></a>

### SGLD


The Stochastic Gradient Langevin Dynamics (SGLD) is based on the paper "Bayesian learning via stochastic gradient langevin dynamics" by Welling and Teh (2011). A link to the article can be found [here](https://dl.acm.org/citation.cfm?id=3104568).


SGLD is an approximation to [Langevin adjusted MH](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm). SGLD uses stochastic gradients that are based on mini-batches of data, and it skips the MH correction step to improve scalability. Computing Metropolis-Hastings accept probabilities requires evaluation likelihoods for the full dataset, making it significantly less scalable. The resulting Gibbs sampler is no longer unbiased since SGLD is an approximate sampler.


```julia
c = sample(model, SGLD(1000, 0.01))
```


```
[SGLD] Finished with
  Running time        = 0.888668504000001;
  Accept rate         = 1.0;
  #lf / sample        = 0.0;
  #evals / sample     = 501.5;
  pre-cond. metric    = [1.0].
```


```julia
plot_sampler(c)
```


![](/assets/figures/samplers_16_1.svg)
