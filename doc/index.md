<a name='optim.dok'></a>
# Optim Package

This package provides a set of optimization algorithms, which all follow
a unified, closure-based API.

This package is fully compatible with the [nn](http://nn.readthedocs.org) package, but can also be
used to optimize arbitrary objective functions.

For now, the following algorithms are provided:

  * [Stochastic Gradient Descent](#optim.sgd)
  * [Averaged Stochastic Gradient Descent](#optim.asgd)
  * [L-BFGS](#optim.lbfgs)
  * [Congugate Gradients](#optim.cg)
  * [AdaDelta](#optim.adadelta)
  * [AdaGrad](#optim.adagrad)
  * [Adam](#optim.adam)
  * [AdaMax](#optim.adamax)
  * [FISTA with backtracking line search](#optim.FistaLS)
  * [Nesterov's Accelerated Gradient method](#optim.nag)
  * [RMSprop](#optim.rmsprop)
  * [Rprop](#optim.rprop)
  * [CMAES](#optim.cmaes)

All these algorithms are designed to support batch optimization as
well as stochastic optimization. It's up to the user to construct an 
objective function that represents the batch, mini-batch, or single sample
on which to evaluate the objective.

Some of these algorithms support a line search, which can be passed as
a function (L-BFGS), whereas others only support a learning rate (SGD).

<a name='optim.overview'></a>
## Overview 

This package contains several optimization routines for [Torch](https://github.com/torch/torch7/blob/master/README.md).
Most optimization algorithms has the following interface:

```lua
x*, {f}, ... = optim.method(opfunc, x, state)
```

where:

* `opfunc`: a user-defined closure that respects this API: `f, df/dx = func(x)`
* `x`: the current parameter vector (a 1D `torch.Tensor`)
* `state`: a table of parameters, and state variables, dependent upon the algorithm
* `x*`: the new parameter vector that minimizes `f, x* = argmin_x f(x)`
* `{f}`: a table of all f values, in the order they've been evaluated (for some simple algorithms, like SGD, `#f == 1`)

<a name='optim.example'></a>
## Example

The state table is used to hold the state of the algorihtm.
It's usually initialized once, by the user, and then passed to the optim function
as a black box. Example:

```lua
state = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i,sample in ipairs(training_samples) do
    local func = function(x)
       -- define eval function
       return f,df_dx
    end
    optim.sgd(func,x,state)
end
```

<a name='optim.algorithms'></a>
## Algorithms

Most algorithms provided rely on a unified interface:
```lua
x_new,fs = optim.method(opfunc, x, state)
```
where: 
x is the trainable/adjustable parameter vector,
state contains both options for the algorithm and the state of the algorihtm,
opfunc is a closure that has the following interface:
```lua
f,df_dx = opfunc(x)
```
x_new is the new parameter vector (after optimization),
fs is a a table containing all the values of the objective, as evaluated during
the optimization procedure: fs[1] is the value before optimization, and fs[#fs]
is the most optimized one (the lowest).

<a name='optim.sgd'></a>
### [x] sgd(opfunc, x, state) 

An implementation of Stochastic Gradient Descent (SGD).

Arguments:

  * `opfunc` : a function that takes a single input (`X`), the point of a evaluation, and returns `f(X)` and `df/dX`
  * `x`      : the initial point
  * `config` : a table with configuration parameters for the optimizer
  * `config.learningRate`      : learning rate
  * `config.learningRateDecay` : learning rate decay
  * `config.weightDecay`       : weight decay
  * `config.weightDecays`      : vector of individual weight decays
  * `config.momentum`          : momentum
  * `config.dampening`         : dampening for momentum
  * `config.nesterov`          : enables Nesterov momentum
  * `state`  : a table describing the state of the optimizer; after each call the state is modified
  * `state.learningRates`      : vector of individual learning rates

Returns :

  * `x`     : the new x vector
  * `f(x)`  : the function, evaluated before the update

<a name='optim.asgd'></a>
### [x] asgd(opfunc, x, state) 

An implementation of Averaged Stochastic Gradient Descent (ASGD): 

```
x = (1 - lambda eta_t) x - eta_t df/dx(z,x)
a = a + mu_t [ x - a ]

eta_t = eta0 / (1 + lambda eta0 t) ^ 0.75
mu_t = 1/max(1,t-t0)
```

Arguments:

  * `opfunc` : a function that takes a single input (`X`), the point of evaluation, and returns `f(X)` and `df/dX`
  * `x` : the initial point
  * `state` : a table describing the state of the optimizer; after each call the state is modified
  * `state.eta0` : learning rate
  * `state.lambda` : decay term
  * `state.alpha` : power for eta update
  * `state.t0` : point at which to start averaging

Returns:

  * `x`     : the new x vector
  * `f(x)`  : the function, evaluated before the update
  * `ax`    : the averaged x vector


<a name='optim.lbfgs'></a>
### [x] lbfgs(opfunc, x, state)

An implementation of L-BFGS that relies on a user-provided line
search function (`state.lineSearch`). If this function is not
provided, then a simple learningRate is used to produce fixed
size steps. Fixed size steps are much less costly than line
searches, and can be useful for stochastic problems.

The learning rate is used even when a line search is provided.
This is also useful for large-scale stochastic problems, where
opfunc is a noisy approximation of `f(x)`. In that case, the learning
rate allows a reduction of confidence in the step size.

Arguments :

  * `opfunc` : a function that takes a single input (`X`), the point of evaluation, and returns `f(X)` and `df/dX`
  * `x` : the initial point
  * `state` : a table describing the state of the optimizer; after each call the state is modified
  * `state.maxIter` : Maximum number of iterations allowed
  * `state.maxEval` : Maximum number of function evaluations
  * `state.tolFun` : Termination tolerance on the first-order optimality
  * `state.tolX` : Termination tol on progress in terms of func/param changes
  * `state.lineSearch` : A line search function
  * `state.learningRate` : If no line search provided, then a fixed step size is used

Returns :
  * `x*` : the new `x` vector, at the optimal point
  * `f`  : a table of all function values: 
   * `f[1]` is the value of the function before any optimization and
   * `f[#f]` is the final fully optimized value, at `x*`


<a name='optim.cg'></a>
### [x] cg(opfunc, x, state)

An implementation of the Conjugate Gradient method which is a rewrite of 
`minimize.m` written by Carl E. Rasmussen. 
It is supposed to produce exactly same results (give
or take numerical accuracy due to some changed order of
operations). You can compare the result on rosenbrock with 
[minimize.m](http://www.gatsby.ucl.ac.uk/~edward/code/minimize/example.html).
```
[x fx c] = minimize([0 0]', 'rosenbrock', -25)
```

Note that we limit the number of function evaluations only, it seems much
more important in practical use.

Arguments :

  * `opfunc` : a function that takes a single input, the point of evaluation.
  * `x`      : the initial point
  * `state` : a table of parameters and temporary allocations.
  * `state.maxEval`     : max number of function evaluations
  * `state.maxIter`     : max number of iterations
  * `state.df[0,1,2,3]` : if you pass torch.Tensor they will be used for temp storage
  * `state.[s,x0]`      : if you pass torch.Tensor they will be used for temp storage

Returns :

  * `x*` : the new x vector, at the optimal point
  * `f`  : a table of all function values where
   * `f[1]` is the value of the function before any optimization and
   * `f[#f]` is the final fully optimized value, at x*

<a name='optim.adadelta'></a>
### [x] adadelta(opfunc, x, config, state)
ADADELTA implementation for SGD http://arxiv.org/abs/1212.5701

Arguments :

* `opfunc` : a function that takes a single input (X), the point of evaluation, and returns f(X) and df/dX
* `x` : the initial point
* `config` : a table of hyper-parameters
* `config.rho` : interpolation parameter
* `config.eps` : for numerical stability
* `state` : a table describing the state of the optimizer; after each call the state is modified
* `state.paramVariance` : vector of temporal variances of parameters
* `state.accDelta` : vector of accummulated delta of gradients

Returns :

* `x` : the new x vector
* `f(x)` : the function, evaluated before the update

<a name='optim.adagrad'></a>
### [x] adagrad(opfunc, x, config, state)
AdaGrad implementation for SGD

Arguments :

* `opfunc` : a function that takes a single input (X), the point of evaluation, and returns f(X) and df/dX
* `x` : the initial point
* `state` : a table describing the state of the optimizer; after each call the state is modified
* `state.learningRate` : learning rate
* `state.paramVariance` : vector of temporal variances of parameters

Returns :

* `x` : the new x vector
* `f(x)` : the function, evaluated before the update

<a name='optim.adam'></a>
### [x] adam(opfunc, x, config, state)
An implementation of Adam from http://arxiv.org/pdf/1412.6980.pdf

Arguments :

* `opfunc` : a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
* `x`      : the initial point
* `config` : a table with configuration parameters for the optimizer
* `config.learningRate`      : learning rate
* `config.beta1`             : first moment coefficient
* `config.beta2`             : second moment coefficient
* `config.epsilon`           : for numerical stability
* `state`                    : a table describing the state of the optimizer; after each call the state is modified

Returns :

* `x`     : the new x vector
* `f(x)`  : the function, evaluated before the update

<a name='optim.adamax'></a>
### [x] adamax(opfunc, x, config, state)
An implementation of AdaMax http://arxiv.org/pdf/1412.6980.pdf

Arguments :

* `opfunc` : a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
* `x`      : the initial point
* `config` : a table with configuration parameters for the optimizer
* `config.learningRate`      : learning rate
* `config.beta1`             : first moment coefficient
* `config.beta2`             : second moment coefficient
* `config.epsilon`           : for numerical stability
* `state`                    : a table describing the state of the optimizer; after each call the state is modified.

Returns :

* `x`     : the new x vector
* `f(x)`  : the function, evaluated before the update

<a name='optim.FistaLS'></a>
### [x] FistaLS(f, g, pl, xinit, params)
FISTA with backtracking line search
* `f`        : smooth function
* `g`        : non-smooth function
* `pl`       : minimizer of intermediate problem Q(x,y)
* `xinit`    : initial point
* `params`   : table of parameters (**optional**)
* `params.L`       : 1/(step size) for ISTA/FISTA iteration (0.1)
* `params.Lstep`   : step size multiplier at each iteration (1.5)
* `params.maxiter` : max number of iterations (50)
* `params.maxline` : max number of line search iterations per iteration (20)
* `params.errthres`: Error thershold for convergence check (1e-4)
* `params.doFistaUpdate` : true : use FISTA, false: use ISTA (true)
* `params.verbose` : store each iteration solution and print detailed info (false)

On output, `params` will contain these additional fields that can be reused.
* `params.L`       : last used L value will be written.

These are temporary storages needed by the algo and if the same params object is 
passed a second time, these same storages will be used without new allocation.
* `params.xkm`     : previous iterarion point
* `params.y`       : fista iteration
* `params.ply`     : ply = pl(y * 1/L grad(f))

Returns the solution x and history of {function evals, number of line search ,...}

Algorithm is published in http://epubs.siam.org/doi/abs/10.1137/080716542

<a name='optim.nag'></a>
### [x] nag(opfunc, x, config, state)      
An implementation of SGD adapted with features of Nesterov's 
Accelerated Gradient method, based on the paper "On the Importance of Initialization and Momentum in Deep Learning" (Sutsveker et. al., ICML 2013).

Arguments :

*  `opfunc` : a function that takes a single input (X), the point of evaluation, and returns f(X) and df/dX
*  `x` : the initial point
*  `state`  : a table describing the state of the optimizer; after each call the state is modified
*  `state.learningRate`      : learning rate
*  `state.learningRateDecay` : learning rate decay
*  `astate.weightDecay`       : weight decay
*  `state.momentum`          : momentum
*  `state.learningRates`     : vector of individual learning rates

Returns :

* `x`     : the new x vector
* `f(x)` : the function, evaluated before the update

<a name='optim.rmsprop'></a>
### [x] rmsprop(opfunc, x, config, state)
An implementation of RMSprop

Arguments :

* `opfunc` : a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
* `x`      : the initial point
* `config` : a table with configuration parameters for the optimizer
* `config.learningRate`      : learning rate
* `config.alpha`             : smoothing constant
* `config.epsilon`           : value with which to initialise m
* `state`                    : a table describing the state of the optimizer; after each call the state is modified
* `state.m`                  : leaky sum of squares of parameter gradients,
* `state.tmp`                : and the square root (with epsilon smoothing)

Returns :

* `x`     : the new x vector
* `f(x)`  : the function, evaluated before the update

<a name='optim.rprop'></a>
### [x] rprop(opfunc, x, config, state)
A plain implementation of Rprop
(Martin Riedmiller, Koray Kavukcuoglu 2013)

Arguments :

* `opfunc` : a function that takes a single input (X), the point of evaluation, and returns f(X) and df/dX
* `x`      : the initial point
* `state`  : a table describing the state of the optimizer; after each call the state is modified
* `state.stepsize`    : initial step size, common to all components
* `state.etaplus`     : multiplicative increase factor, > 1 (default 1.2)
* `state.etaminus`    : multiplicative decrease factor, < 1 (default 0.5)
* `state.stepsizemax` : maximum stepsize allowed (default 50)
* `state.stepsizemin` : minimum stepsize allowed (default 1e-6)
* `state.niter`       : number of iterations (default 1)

Returns :

* `x`     : the new x vector
* `f(x)`  : the function, evaluated before the update



 
<a name='optim.cmaes'></a>
### [x] cmaes(opfunc, x, config, state)
An implementation of `CMAES` (Covariance Matrix Adaptation Evolution Strategy), 
ported from https://www.lri.fr/~hansen/barecmaes2.html.

CMAES is a stochastic, derivative-free method for heuristic global optimization of non-linear or non-convex continuous optimization problems. Note that this method will on average take much more function evaluations to converge then a gradient based method.

Arguments:

* `opfunc` : a function that takes a single input (X), the point of evaluation, and returns f(X) and df/dX. Note that df/dX is not used and can be left 0
* `x` : the initial point
* `state.sigma`     : float, initial step-size (standard deviation in each coordinate)
* `state.maxEval`   : int, maximal number of function evaluations
* `state.ftarget`   : float, target function value
* `state.popsize`   : population size. If this is left empty, 4 + int(3 * log(|x|)) will be used
* `state.ftarget`   : stop if fitness < ftarget
* `state.verb_disp` : display info on console every verb_disp iteration, 0 for never

Returns:
* `x*` : the new `x` vector, at the optimal point
* `f`  : a table of all function values: 
  * `f[1]` is the value of the function before any optimization and
  * `f[#f]` is the final fully optimized value, at `x*`
