----------------------------------------------------------------------
-- An implementation of L-BFGS, heavily inspired from minFunc.
--
-- For now, we only implement one type of line search:
-- Bracketing w/ Cubic Interpolation/Extrapolation 
-- with function + gradient values (Wolfe Criterion)
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.maxIter     :  Maximum number of iterations allowed
--   state.maxEval     :  Maximum number of function evaluations
--   state.tolFun      :  Termination tolerance on the first-order optimality
--   state.tolX        :  Termination tol on progress in terms of func/param changes
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function value, at the optimal point
--
function optim.lbfgs(opfunc, x, state)
   -- get/update state
   local state = state or {}
   local maxIter = state.maxIter or 20
   local maxEval = state.maxEval or 40
   local tolFun = state.tolFun or 1e-5
   local tolX = state.tolFun or 1e-9
   local nCorrection = state.nCorrection or 100
   local c1 = state.lineSearchDecrease or 1e-4
   local c2 = state.lineSearchCurvature or 0.9
   local state.funcEval = state.funcEval or 0

   -- lab -> local
   local zeros = lab.zeros
   local randn = lab.randn

   -- initial step length
   local t = 1

   -- evaluate initial f(x) and df/dx
   local f,g = opfunc(x)
   local currentFuncEval = 1
   state.funcEval = state.funcEval + 1

   -- check optimality of initial point
   if g:abs():sum() <= tolFun then
      -- optimality condition below tolFun
      return x,f
   end

   -- optimize for a max of maxIter iterations
   local nIter = 0
   while nIter < maxIter do
      -- keep track of nb of iterations
      nIter = nIter + 1

      ------------------------------------------------------------
      -- computer gradient descent direction
      ------------------------------------------------------------
      local d,old_dirs,old_stps,Hdiag,g_old
      if i == 1 then
         d = -g
         old_dirs = {zeros(g:size())}
         old_stps = {zeros(d:size())}
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
                  table.insert(old_dirs, prev_old_dirs[i])
                  table.insert(old_stps, prev_old_stps[i])
               end
            end
            
            -- store new direction/step
            table.insert(old_dirs, s)
            table.insert(old_stps, y)

            -- update scale of initial Hessian approximation
            Hdiag = ys/(y*y)
         end

         -- compute the approximate (L-BFGS) inverse Hessian 
         -- multiplied by the gradient
         local p = g:size()
         local k = #old_dirs

         local ro = {}
         for i = 1,k do
            ro[i] = 1 / (old_stps[i] * old_dirs[i])
         end

         local q = zeros(k+1,p)
         local r = zeros(k+1,p)
         local al = zeros(k)
         local be = zeros(k)

         q[k+1] = -g

         for i = k,1,-1 do
            al[i] = ro[i] * old_dirs[i] * q[i+1]
            q[i] = q[i+1] - al[i] * old_stps[i]
         end

         -- multiply by initial Hessian
         r[1] = Hdiag * q[1]

         for i = 1,k do
            be[i] = ro[i] * old_stps[i] * r[i]
            r[i+1] = r[i] + old_dirs[i] * (al[i] - be[i])
         end

         -- final direction:
         d = r[k+1]
      end
      g_old = g:clone()
      f_old = f:clone()

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
      t = 1

      -- perform line search, satisfying Wolfe condition
      --[t,f,g,lsFuncEval] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS=4,25,tolX,false,false,1,opfunc)

      -- update func eval
      currentFuncEval = currentFuncEval + lsFuncEval
      state.funcEval = state.funcEval + lsFuncEval
      
      -- update parameters
      x = x + d*t

      ------------------------------------------------------------
      -- check conditions
      ------------------------------------------------------------
      if (d*t):abs():sum() <= tolX then
         -- step size below tolX
         break
      end

      if (f-f_old):abs() < tolX then
         -- function value changing less than tolX
         break
      end

      if currentFuncEval >= state.maxEval then
         -- max nb of function evals
         break
      end
   end

   -- return f(x_new), x_new
   return x,f
end
