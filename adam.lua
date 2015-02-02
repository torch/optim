--[[ An implementation of Adam http://arxiv.org/pdf/1412.6980v2.pdf

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- 'config.epsilon'           : for numerical stability
- 'config.lambda'            : first moment decay
- 'state = {t, m, v}'        : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.adam(opfunc, x, config, state)
    -- get parameters
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 2e-6

    local beta1 = config.beta1 or 0.1
    local beta2 = config.beta2 or 0.001
    local epsilon = config.epsilon or 10e-8
    local lambda = config.lambda or 10e-8

    local fx, dfdx = opfunc(x)

    state.t = state.t or 1
    state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
    state.v = state.v or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)

    local bt1 = 1 - (1-beta1)*torch.pow(lambda,state.t-1)
    state.m = torch.add(torch.mul(dfdx, bt1), torch.mul(state.m, 1-bt1))
    state.v = torch.add(torch.mul(torch.pow(dfdx, 2), beta2), torch.mul(state.v, 1-beta2))

    local update = torch.cmul(state.m, torch.pow(torch.add(torch.pow(state.v, 2), epsilon),-1))
    update:mul(lr * torch.sqrt(1-torch.pow((1-beta2),2)) * torch.pow(1-torch.pow((1-beta1),2), -1))

    x:add(-update)
    state.t = state.t + 1

    -- return x*, f(x) before optimization
    return x,{fx}, update
end
