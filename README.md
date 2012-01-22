# optim: an optimization package for Torch7

## Requirements

* Torch7 (github.com/andresy/torch)

## Installation

* Install Torch7 (refer to its own documentation).
* clone this project into dev directory of Torch7.
* Rebuild torch, it will include new projects too.

## Info

This package contains several optimization routines for Torch7.

Each optimization algorithm is based on the same interface:
x*, {f}, ... = optim.method(func, x, state)

with:
* func  : a user-defined closure that respects this API: f,df/dx = func(x)
* x     : the current parameter vector (a 1d torch tensor)
* state : a table of parameters, and state variables, dependent upon the algorithm
* x*    : the new parameter vector that minimizes f, x* = argmin_x f(x)
* {f}   : a table of all f values, in the order they've been evaluated
          (for some simple algorithms, like SGD, #f == 1)
