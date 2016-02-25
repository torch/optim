# Optimization package

This package contains several optimization routines for [Torch](https://github.com/torch/torch7/blob/master/README.md).
Each optimization algorithm is based on the same interface:

```lua
x*, {f}, ... = optim.method(func, x, state)
```

where:

* `func`: a user-defined closure that respects this API: `f, df/dx = func(x)`
* `x`: the current parameter vector (a 1D `torch.Tensor`)
* `state`: a table of parameters, and state variables, dependent upon the algorithm
* `x*`: the new parameter vector that minimizes `f, x* = argmin_x f(x)`
* `{f}`: a table of all f values, in the order they've been evaluated
  (for some simple algorithms, like SGD, `#f == 1`)

## Available algorithms

Please check [this file](doc/index.md) for the full list of
optimization algorithms available and examples. Get also into the
[`test`](test/) directory for straightforward examples using the
[Rosenbrock's](test/rosenbrock.lua) function.

## Important Note

The state table is used to hold the state of the algorithm.
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
