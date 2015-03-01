--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to inistialise m
- 'config.epsilon2'          : stablisation to prevent mean square going to zero
- 'config.max_gain'          : stabilisation to prevent lr multiplier exploding
- 'config.min_gain'          : stabilisation to prevent lr multiplier exploding
- 'state = {m}'              : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.rmsprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local alpha = config.alpha or 0.998
    local epsilon = config.epsilon or 1e-8
    local epsilon2 = config.epsilon2 or 1e-4
    local max_gain = config.max_gain or 100
    local min_gain = config.min_gain or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) initialize mean square values
    state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(epsilon)

    -- (3) calculate new mean squared value
    local dfdx_sq = dfdx:clone()
    local dfdx_sq = dfdx_sq:cmul(dfdx_sq)
    state.m:mul(alpha)
    state.m:add(dfdx_sq:mul(1.0-alpha))
    state.m:add(epsilon2)

    -- (4) perform update
    local one_over_rms = torch.pow(state.m, -0.5)
    one_over_rms:clamp(min_gain, max_gain)
    local update = torch.cmul(torch.mul(dfdx,lr), one_over_rms)
    x:add(-update)

    -- return x*, f(x) before optimization
    return x, {fx}, update
end

