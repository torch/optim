----------------------------------------------------------------------
-- An implementation of ASGD
--
-- ASGD: 
--     x := (1 - lambda eta_t) x - eta_t df/dx(z,x)
--     a := a + mu_t [ x - a ]
--
--  eta_t = eta_0 / (1 + lambda eta0 t) ^ 0.75
--   mu_t = 1/max(1,t-t0)
-- 
-- implements ASGD algoritm as in L.Bottou's sgd-2.0
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.eta0/learningRate : learning rate
--   state.lambda            : decay term
--   state.alpha             : power for eta update
--   state.t0                : point at which to start averaging
--   state.learningRates     : vector of individual learning rates
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
-- ax    : the averaged x vector
--
function optim.asgd(opfunc, x, state)
   -- (0) get/update state
   local state = state or {}
   local eta_t = state.eta_t or eta_0
   local lambda = state.lambda or 1
   local alpha = state.alpha or 1
   local t0 = state.t0 or 1e6
   local lrs = state.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounters

   state.eta_0 = state.eta0 or state.learningRate or 1e-4
   state.mu_0 = state.mu_t or 0
   state.t = state.t or 0

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) decay term
   x:mul(1 - lambda*state.eta_t)

   -- (3) update x
   if lrs then
      if not state.deltax then
         state.deltax = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltax:copy(lrs):cmul(dfdx)
      x:add(-state.eta_t, state.deltax)
   else
      x:add(-state.eta_t, state.dfdx)
   end

   -- (4) averaging
   state.ax = state.a or torch.Tensor():typeAs(x):resizeAs(x):zero()
   state.tmp = state.tmp or torch.Tensor():typeAs(state.ax):resizeAs(state.ax)
   if state.mu_t ~= 1 then
      state.tmp:copy(x)
      state.tmp:add(-1,state.ax):mul(state.mu_t)
      state.ax:add(state.tmp)
   else
      state.ax:copy(x)
   end

   -- (5) update eta_t and mu_t
   state.t = state.t + 1
   state.eta_t = state.eta0 / math.pow((1 + state.lambda * state.eta0 * state.t), 0.75)
   state.mu_t = 1 / math.max(1, state.t - state.t0)

   -- return f(x_old), x_new, and averaged x
   return x,fx,state.ax
end
