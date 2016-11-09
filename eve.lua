--[[ EVE implementation https://arxiv.org/pdf/1611.01505v1.pdf

ARGS:
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
    local state = state or {}
    local lr = config.learningRate or 1e-3
    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local eps = config.epsilon or 1e-8
    local beta3 = config.beta3 or 0.999
    local thl = config.thl or 0.1
    local thu = config.thu or 10
    state.d = state.d or 1
    state.t = state.t or 0
    state.fhat = state.fhat or 0
    -- (2) evaluate f(x) and df/dx


    local fx, dfdx = opfunc(x)

    state.m = state.m or x.new(dfdx:size()):zero()
    state.m:mul(beta1):add(1-beta1, dfdx)

    state.v = state.v or x.new(dfdx:size()):zero()
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

    state.denom = state.denom or x.new(dfdx:size()):zero()
    state.denom:copy(state.v)
    -- update
    state.t = state.t + 1

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
        state.d = beta3 * state.d + (1 - beta3) * r
    end


    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local alpha = lr * state.d / math.sqrt(biasCorrection2) * biasCorrection1
    state.denom:sqrt():add(eps)
    x:addcdiv(-alpha, state.m, state.denom)

    -- return x*, f(x) before optimization
    return x, {fx}
end
