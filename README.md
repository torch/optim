Optim: an optimization package for Torch7
=========================================

Requirements
------------

* Torch7 (www.torch.ch)

Installation
------------

* Install Torch7 (refer to its own documentation).
* Use `torch-pkg` to install optim:

```
torch-pkg install optim
```

or from these sources:

```
cd optim;
torch-pkg deploy
```

Info
----

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

Important Note: the state table is used to hold the state of the algorihtm.
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
    optim.sgd(f,x,state)
end
```
