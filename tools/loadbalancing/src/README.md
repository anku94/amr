# Load Balancing Policies

```
- benchmarks
- blocksim
- clustersim
- distrib
- lb
- policies
- scaling
```

## Distributions

Config file path is specified via the environment variable `DISTRIB_CONFIG_FPATH`.

Parameters:

```
distribution: string from [gaussian, exp, powerlaw]
N_min: int [minimum value of the generated distrib]
N_max: int [minimum value of the generated distrib]
gaussian_mean: double
gaussian_std: double
exp_lambda: double
powerlaw_alpha: double
```

Example config (also defaults):

```
distribution=gaussian
# comments are supported
N_min=1
N_max=100
gaussian_mean=10
gaussian_std=0.5
exp_lambda=0.1
powerlaw_alpha=-3.0
```
