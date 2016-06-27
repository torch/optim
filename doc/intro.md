<a name='optim.overview'></a>
## Overview

Most optimization algorithms have the following interface:

```lua
x*, {f}, ... = optim.method(opfunc, x, state)
```

where:

* `opfunc`: a user-defined closure that respects this API: `f, df/dx = func(x)`
* `x`: the current parameter vector (a 1D `Tensor`)
* `state`: a table of parameters, and state variables, dependent upon the algorithm
* `x*`: the new parameter vector that minimizes `f, x* = argmin_x f(x)`
* `{f}`: a table of all `f` values, in the order they've been evaluated (for some simple algorithms, like SGD, `#f == 1`)


<a name='optim.example'></a>
## Example

The state table is used to hold the state of the algorihtm.
It's usually initialized once, by the user, and then passed to the optim function as a black box.
Example:

```lua
state = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i, sample in ipairs(training_samples) do
    local func = function(x)
       -- define eval function
       return f, df_dx
    end
    optim.sgd(func, x, state)
end
```

