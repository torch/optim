====== Optimization Package =======
{{anchor:optim.dok}}

This package provides a set of optimization algorithms, which all follow
a unified, closure-based API.

This package is fully compatible with the 'nn' package, but can also be
used to optimize arbitrary objective functions.

For now, the following algorithms are provided:
  * Stochastic Gradient Descent (SGD): [[#optim.sgd|optim.sgd]]
  * Averaged Stochastic Gradient Descent (ASGD): [[#optim.asgd|optim.asgd]]
  * L-BFGS: [[#optim.lbfgs|optim.lbfgs]]
  * Congugate Gradients (CG): [[#optim.cg|optim.cg]]

All these algorithms are designed to support batch optimization as
well as stochastic optimization. It's up to the user to construct an 
objective function that represents the batch, mini-batch, or single sample
on which to evaluate the objective.

Some of these algorithms support a line search, which can be passed as
a function (L-BFGS), whereas others only support a learning rate (SGD).


====== Overview of the Optimization Package ======
{{anchor:optim.overview.dok}}

Rather than long descriptions, let's simply start with a little example.

<file lua>
-- write an example here.
</file>

===== Simple Objective =====

===== Neural Network Objective =====


====== Algorithms ======
{{anchor:nn.API}}

All the algorithms provided rely on a unified interface:
<file lua>
w_new,fs = optim.method(func,w,state)
</file>
where: 
w is the trainable/adjustable parameter vector,
state contains both options for the algorithm and the state of the algorihtm,
func is a closure that has the following interface:
<file lua>
f,df_dw = func(w)
</file>
w_new is the new parameter vector (after optimization),
fs is a a table containing all the values of the objective, as evaluated during
the optimization procedure: fs[1] is the value before optimization, and fs[#fs]
is the most optimized one (the lowest).

===== [x] sgd(func, w, state) =====
{{anchor:optim.sgd}}

An implementation of Stochastic Gradient Descent.

===== [x] asgd(func, w, state) =====
{{anchor:optim.asgd}}

An implementation of Averaged Stochastic Gradient Descent.

===== [x] lbfgs(func, w, state) =====
{{anchor:optim.lbfgs}}

An implementation of L-BFGS.

===== [x] cg(func, w, state) =====
{{anchor:optim.cg}}

An implementation of the Conjugate Gradient method.


