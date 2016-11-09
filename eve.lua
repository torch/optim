--[[ EVE implementation https://arxiv.org/pdf/1611.01505v1.pdf

ARGS:
- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- `config.learningRateDecay` : learning rate decay
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- `config.beta3`             : exponential decay rate for relative change
- 'config.epsilon'           : for numerical stability
- `config.thl`               : lowerbound threshold
- `config.thu`               : upperbound threshold
- 'config.weightDecay'       : weight decay
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
RETURN:
- `x` : the new x vector
- `f(x)` : the function, evaluated before update
]]
function optim.eve(opfunc, x, config, state)
    -- (0) get/update state
    if config == nil and state == nil then
        print('no state table, EVE initializing')
    end

    local config = config or {}
    local state  = state or {}

    local lr     = config.learningRate or 1e-3
    local lrd    = config.learningRateDecay or 0
    local beta1  = config.beta1 or 0.9
    local beta2  = config.beta2 or 0.999
    local beta3  = config.beta3 or 0.999
    local eps    = config.epsilon or 1e-8
    local thl    = config.thl or 0.1
    local thu    = config.thu or 10
    local wd     = config.weightDecay or 0

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) weight decay
    if wd ~= 0 then
      dfdx:add(wd, x)
    end

    -- Initialize state
    state.d = state.d or 1
    state.t = state.t or 0
    state.fhat = state.fhat or 0

    -- (3) learning rate decay (annealing)
    local clr = lr / (1 + state.t*lrd)
    state.t = state.t + 1

    -- Decay the first and second moment running average coefficient
    state.m = state.m or x.new(dfdx:size()):zero()
    state.m:mul(beta1):add(1-beta1, dfdx)

    state.v = state.v or x.new(dfdx:size()):zero()
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

    state.denom = state.denom or x.new(dfdx:size()):zero()
    state.denom:copy(state.v)

    if state.t == 1 then
        state.d = 1
        state.fhat = fx
        state.t = 1
    else
        local l, u = 0, 0 -- lowerbound and upperbound
        if fx > state.fhat then
            l, u = thl + 1, thu + 1
        else
            l, u = 1 / (thu+1), 1 / (thl + 1)
        end
        local fhat = state.fhat * math.min(math.max(l, fx / state.fhat), u)
        local r = math.abs(fhat - state.fhat) / math.min(fhat, state.fhat)
        state.fhat = fhat
        -- Decay the relative change
        state.d = beta3 * state.d + (1 - beta3) * r
    end


    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local stepSize = clr * state.d/math.sqrt(biasCorrection2) * biasCorrection1
    state.denom:sqrt():add(eps)

    -- (4) update x
    x:addcdiv(-stepSize, state.m, state.denom)

    -- return x*, f(x) before optimization
    return x, {fx}
end
