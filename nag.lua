----------------------------------------------------------------------
-- An implementation of SGD adapted with features of Nesterov's 
-- Accelerated Gradient method, based on the paper
-- On the Importance of Initialization and Momentum in Deep Learning
-- Sutsveker et. al., ICML 2013
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.learningRate      : learning rate
--   state.learningRateDecay : learning rate decay
--   state.weightDecay       : weight decay
--   state.momentum          : momentum
--   state.learningRates     : vector of individual learning rates
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
--
-- (Dilip Krishnan, 2013)
--

function optim.nag(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local lrs = config.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   -- first step in the direction of the momentum vector
   if not state.x_copy then
      state.x_copy = x:clone()
   else
      state.x_copy:resizeAs(x):copy(x)
   end
   if state.dfdx then
      x:add(mom, state.dfdx) 
   end
   -- then compute gradient at that point
   -- comment out the above line to get the original SGD
   local fx,dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- (4) apply momentum
   if not state.dfdx then
      state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   end

   if mom ~= 0 then
      state.dfdx:mul(mom)
   end

   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      state.dfdx:add(-clr, state.deltaParameters)
   else
      state.dfdx:add(-clr, dfdx)
   end
   state.x_copy:add(state.dfdx)
   x:copy(state.x_copy)

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x, f(x) before optimization
   return x,{fx}
end
