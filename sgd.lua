----------------------------------------------------------------------
-- A plain implementation of SGD
--
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
function optim.sgd(opfunc, x, state)
   -- (0) get/update state
   local state = state or {}
   local lr = state.learningRate or 1e-3
   local lrd = state.learningRateDecay or 0
   local wd = state.weightDecay or 0
   local mom = state.momentum or 0
   local lrs = state.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounters

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-mom, dfdx)
      end
      dfdx = state.dfdx
   end

   -- (2) weight decay
   if wd ~= 0 then
      x:add(-wd, x)
   end

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
      
   -- (4) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      x:add(-clr, state.dfdx)
   end

   -- (5) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return f(x_old), and x_new
   return x,fx
end
