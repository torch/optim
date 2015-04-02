--[[ A plain implementation of ADADELTA 
    http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    ARGS:

    - `opfunc` : a function that takes a single input (X), the point
    of a evaluation, and returns f(X) and df/dX
    - `x`      : the initial point
    - `config` : a table with configuration parameters for the optimizer
    - `config.decayRate`           : decay rate
    - `config.offset`              : offset
    - `state`  : a table describing the state of the optimizer; after each
    call the state is modified
    - `state.accSquaredDeltaWithDecay`
               : vector of accumulated squared deltas from previous steps
    with corresponding decay rate
    - `state.accSquaredGradsWithDecay`
               : vector of accumulated squared gradients from previous steps
    with corresponding decay rate
    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update
]]
function optim.adadelta(opfunc, x, config, state)
    local config = config or {}
    local state = state or config
    local decay = config.decayRate or 0.9
    local eps = config.offset or 1e-6

    assert(decay >= 0 and decay <= 1, "Wrong value for decay rate: " .. decay)
    assert(eps >= 0, "Wrong value for offset constant: " .. eps)

    local fx, dfdx = opfunc(x)

    if not state.accSquaredDeltaWithDecay then
        state.accSquaredDeltaWithDecay =
            torch.Tensor():typeAs(x):resizeAs(x):zero()
        state.accSquaredGradsWithDecay =
            torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
    end

    state.accSquaredGradsWithDecay:mul(decay):addcmul(1.0 - decay, dfdx, dfdx)

    if not state.prev_rms_delta then
        state.prev_rms_delta = torch.Tensor()
            :typeAs(state.accSquaredDeltaWithDecay)
            :resizeAs(state.accSquaredDeltaWithDecay)
        state.rms_grad = torch.Tensor()
            :typeAs(state.accSquaredGradsWithDecay)
            :resizeAs(state.accSquaredGradsWithDecay)
    end

    state.prev_rms_delta:copy(state.accSquaredDeltaWithDecay):add(eps):sqrt()
    state.rms_grad:copy(state.accSquaredGradsWithDecay):add(eps):sqrt()    
    local delta = state.prev_rms_delta:cdiv(state.rms_grad):cmul(dfdx):mul(-1.0)

    state.accSquaredDeltaWithDecay:mul(decay):addcmul(1.0 - decay, delta, delta)

    x:add(delta)

    return x, {fx}
end
