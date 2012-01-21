----------------------------------------------------------------------
-- An implementation of L-BFGS, heavily inspired from minFunc.
--
-- This implementation of L-BFGS relies on a user-provided line
-- search function (state.lineSearch). If this function is not
-- provided, then a simple learningRate is used to produce fixed
-- size steps. Fixed size steps are much less costly than line
-- searches, and can be useful for stochastic problems.
--
-- The learning rate is used even when a line search is provided.
-- This is also useful for large-scale stochastic problems, where
-- opfunc is a noisy approximation of f(x). In that case, the learning
-- rate allows a reduction of confidence in the step size.
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.maxIter      :  Maximum number of iterations allowed
--   state.maxEval      :  Maximum number of function evaluations
--   state.tolFun       :  Termination tolerance on the first-order optimality
--   state.tolX         :  Termination tol on progress in terms of func/param changes
--   state.lineSearch   :  A line search function
--   state.learningRate : If no line search provided, then a fixed step size is used
--
-- RETURN:
-- x* : the new x vector, at the optimal point
-- f  : a table of all function values: 
--      f[1] is the value of the function before any optimization
--      f[#f] is the final fully optimized value, at x*
--
-- (Clement Farabet, 2012)
--
function optim.lbfgs(opfunc, x, state)
   -- get/update state
   local state = state or {}
   local maxIter = tonumber(state.maxIter) or 20
   local maxEval = tonumber(state.maxEval) or maxIter*1.25
   local tolFun = state.tolFun or 1e-5
   local tolX = state.tolX or 1e-9
   local nCorrection = state.nCorrection or 100
   local lineSearch = state.lineSearch
   local c1 = state.lineSearchDecrease or 1e-4
   local c2 = state.lineSearchCurvature or 0.9
   local learningRate = state.learningRate or 1
   local verbose = state.verbose or false
   state.funcEval = state.funcEval or 0

   -- verbose function
   local function verbose(...)
      if verbose then print('<optim.lbfgs> ', ...) end
   end

   -- import some functions
   local zeros = lab.zeros
   local randn = lab.randn
   local append = table.insert
   local abs = math.abs
   local min = math.min

   -- evaluate initial f(x) and df/dx
   local f,g = opfunc(x)
   local f_hist = {f}
   local currentFuncEval = 1
   state.funcEval = state.funcEval + 1

   -- check optimality of initial point
   if g:clone():abs():sum() <= tolFun then
      -- optimality condition below tolFun
      verbose('optimality condition below tolFun')
      return x,f
   end

   -- optimize for a max of maxIter iterations
   local nIter = 0
   local d,old_dirs,old_stps,Hdiag,g_old,f_old,t
   while nIter < maxIter do
      -- keep track of nb of iterations
      nIter = nIter + 1

      ------------------------------------------------------------
      -- computer gradient descent direction
      ------------------------------------------------------------
      if nIter == 1 then
         d = -g
         old_dirs = {}
         old_stps = {}
         Hdiag = 1
      else
         -- do lbfgs update (update memory)
         local y = g - g_old
         local s = d*t
         local ys = y*s
         if ys > 1e-10 then
            -- updating memory
            if #old_dirs == nCorrection then
               -- shift history by one (limited-memory)
               local prev_old_dirs = old_dirs
               local prev_old_stps = old_stps
               old_dirs = {}
               old_stps = {}
               for i = 2,#prev_old_dirs do
                  append(old_dirs, prev_old_dirs[i])
                  append(old_stps, prev_old_stps[i])
               end
            end
            
            -- store new direction/step
            append(old_dirs, s)
            append(old_stps, y)

            -- update scale of initial Hessian approximation
            Hdiag = ys/(y*y)
         end

         -- compute the approximate (L-BFGS) inverse Hessian 
         -- multiplied by the gradient
         local p = g:size(1)
         local k = #old_dirs

         local ro = zeros(k)
         for i = 1,k do
            ro[i] = 1 / (old_stps[i] * old_dirs[i])
         end

         local q = zeros(k+1,p)
         local r = zeros(k+1,p)
         local al = zeros(k)
         local be = zeros(k)

         q[k+1] = -g

         for i = k,1,-1 do
            al[i] = old_dirs[i] * q[i+1] * ro[i]
            q[i] = q[i+1] - old_stps[i] * al[i]
         end

         -- multiply by initial Hessian
         r[1] = q[1] * Hdiag

         for i = 1,k do
            be[i] = old_stps[i] * r[i] * ro[i]
            r[i+1] = r[i] + old_dirs[i] * (al[i] - be[i])
         end

         -- final direction:
         d = r[k+1]
      end
      g_old = g:clone()
      f_old = f

      ------------------------------------------------------------
      -- compute step length
      ------------------------------------------------------------
      -- directional derivative
      local gtd = g * d

      -- check that progress can be made along that direction
      if gtd > -tolX then
         break
      end

      -- reset initial guess for step size
      if nIter == 1 then
         t = min(1,1/g:clone():abs():sum()) * learningRate
      else
         t = learningRate
      end

      -- optional line search: user function
      local lsFuncEval = 0
      if lineSearch and type(lineSearch) == 'function' then
         -- perform line search, using user function
         f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,c1,c2,tolX)
         append(f_hist, f)
      else
         -- no line search, simply move with fixed-step and re-evaluate f(x)
         x:add(t,d)
         f,g = opfunc(x)
         lsFuncEval = 1
         append(f_hist, f)
      end

      -- update func eval
      currentFuncEval = currentFuncEval + lsFuncEval
      state.funcEval = state.funcEval + lsFuncEval

      ------------------------------------------------------------
      -- check conditions
      ------------------------------------------------------------
      if g:clone():abs():sum() <= tolFun then
         -- check optimality
         verbose('optimality condition below tolFun')
         break
      end

      if (d*t):abs():sum() <= tolX then
         -- step size below tolX
         verbose('step size below tolX')
         break
      end

      if abs(f-f_old) < tolX then
         -- function value changing less than tolX
         verbose('function value changing less than tolX')
         break
      end

      if currentFuncEval >= maxEval then
         -- max nb of function evals
         verbose('max nb of function evals')
         break
      end
   end

   -- return optimal x, and history of f(x)
   return x,f_hist,currentFuncEval
end
