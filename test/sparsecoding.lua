require 'kex'

-- L1 FISTA Solution
-- L1 solution with a linear dictionary ||Ax-b||^2 + \lambda ||x||_1
-- D     : dictionary, each column is a dictionary element
-- params: set of params to pass to FISTA and possibly temp allocation (**optional**)
--         check unsup.FistaLS function for details.
-- returns fista : a table with the following entries
-- fista.run(x,lambda) : run L1 sparse coding algorithm with input x and lambda.
-- The following entries will be allocated and reused by each call to fista.run(x,lambda)
-- fista.reconstruction: reconstructed input.
-- fista.gradf         : gradient of L2 part of the problem wrt x
-- fista.code          : the solution of L1 problem
-- The following entries just point to data passed to fista.run(x)
-- fista.input         : points to the tensor 'x' used in the last fista.run(x,lambda)
-- fista.lambda        : the lambda value used in the last fista.run(x,lambda)
function optim.FistaL1(D, params)

   -- this is for keeping parameters related to fista algorithm
   local params = params or {}
   -- this is for temporary variables and such
   local fista = {}

   -- related to FISTA
   params.L = params.L or 0.1
   params.Lstep = params.Lstep or 1.5
   params.maxiter = params.maxiter or 50
   params.maxline = params.maxline or 20
   params.errthres = params.errthres or 1e-4
   
   -- temporary stuff that might be good to keep around
   fista.reconstruction = torch.Tensor()
   fista.gradf = torch.Tensor()
   fista.gradg = torch.Tensor()
   fista.code = torch.Tensor()

   -- these will be assigned in run(x)
   -- fista.input points to the last input that was run
   -- fista.lambda is the lambda value from the last run
   fista.input = nil
   fista.lambda = nil

   -- CREATE FUNCTION CLOSURES
   -- smooth function
   fista.f = function (x,mode)

		local reconstruction = fista.reconstruction
		local input = fista.input
		-- -------------------
		-- function evaluation
		if x:dim() == 1 then
		   --print(D:size(),x:size())
		   reconstruction:resize(D:size(1))
		   reconstruction:addmv(0,1,D,x)
		elseif x:dim(2) then
		   reconstruction:resize(x:size(1),D:size(1))
		   reconstruction:addmm(0,1,x,D:t())
		end
		local fval = input:dist(reconstruction)^2
		
		-- ----------------------
		-- derivative calculation
		if mode and mode:match('dx') then
		   local gradf = fista.gradf
		   reconstruction:add(-1,input):mul(2)
		   gradf:resizeAs(x)
		   if input:dim() == 1 then
		      gradf:addmv(0,1,D:t(),reconstruction)
		   else
		      gradf:addmm(0,1,reconstruction, D)
		   end
		   ---------------------------------------
		   -- return function value and derivative
		   return fval, gradf, reconstruction
		end
		
		------------------------
		-- return function value
		return fval, reconstruction
	     end

   -- non-smooth function L1
   fista.g =  function (x)

		 local fval = fista.lambda*x:norm(1)

		 if mod and mode:match('dx') then
		    local gradg = fista.gradg
		    gradg:resizAs(x)
		    gradg:sign():mul(fista.lambda)
		    return fval,gradg
		 end
		 return fval
	      end
   
   -- argmin_x Q(x,y), just shrinkage for L1
   fista.pl = function (x,L)
		 x:shrinkage(fista.lambda/L)
	      end
   
   fista.run = function(x, lam, codeinit)
		  local code = fista.code
		  fista.input = x
		  fista.lambda = lam
		  
		  -- resize code, maybe a different number of dimensions
		  -- fill with zeros, initial point
		  if codeinit then
		     code:resizeAs(codeinit)
		     code:copy(codeinit)
		  else
		     if x:dim() == 1 then
			code:resize(D:size(2))
		     elseif x:dim() == 2 then
			code:resize(x:size(1),D:size(2))
		     else
			error(' I do not know how to handle ' .. x:dim() .. ' dimensional input')
		     end
		     code:fill(0)
		  end
		  -- return the result of unsup.FistaLS call.
		  return optim.FistaLS(fista.f, fista.g, fista.pl, fista.code, params)
	       end

   return fista
end

