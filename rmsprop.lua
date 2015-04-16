--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to inistialise m
- 'state = {m, dfdx_sq}'     : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.rmsprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.95
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) initialize mean square values and square gradient storage
    state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
    state.tmp = state.tmp or x.new(dfdx:size()):zero()

    -- (3) calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha,dfdx,dfdx)

    -- (4) perform update
    state.tmp:sqrt(state.m)
    x:addcdiv(-lr, dfdx, state.tmp:add(epsilon))

    -- return x*, f(x) before optimization
    return x, {fx}, state.tmp
end

