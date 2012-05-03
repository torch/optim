local function isreal(x)
   return x == x
end

local function isnan(x)
   return not x == x
end

local function roots(c)
   local n = c:size(1)-1
   local A = torch.diag(torch.ones(n-1),-1)
   A[1] = -c[{ {2,n+1} }]/c[1];
   local e = torch.eig(A,'N')
   return e
end

local function real(x)
   if type(x) == number then return x end
   return x[{ {} , 1}]
end

local function imag(x)
   if type(x) == 'number' then return 0 end
   if x:nDimension() == 1 then
      return torch.zeros(x:size(1))
   else
      return x[{ {},  2}]
   end
end

local function polyval(p,x)
   if type(x) == 'number' then x=torch.Tensor{x} end
   local val = x.new(x:size(1))
   local pwr = p:size(1)
   p:apply(function(pc) pwr = pwr-1; val:add(pc,torch.pow(x,pwr)); return pc end)
   return val
end

----------------------------------------------------------------------
-- Minimum of interpolating polynomial based on function and 
-- derivative values
--
-- ARGS:
-- points : N triplets (x,f,g), must be a Tensor
-- xmin   : min value that brackets minimum (default: min of points)
-- xmax   : max value that brackets maximum (default: max of points)
--
-- RETURN:
-- minPos : position of minimum
--
function optim.polyinterp(points,xminBound,xmaxBound)
   -- locals
   local sqrt = torch.sqrt
   local mean = torch.mean
   local Tensor = torch.Tensor
   local zeros = torch.zeros
   local max = math.max
   local min = math.min

   -- nb of points / order of polynomial
   local nPoints = points:size(1)
   local order = nPoints*2-1

   -- returned values
   local minPos

   -- Code for most common case:
   --   + cubic interpolation of 2 points w/ function and derivative values for both
   --   + no xminBound/xmaxBound
   if nPoints == 2 and order == 3 and not xminBound and not xmaxBound then
      -- Solution in this case (where x2 is the farthest point):
      --    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
      --    d2 = sqrt(d1^2 - g1*g2);
      --    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
      --    t_new = min(max(minPos,x1),x2);
      local minVal,minPos = points[{ {},1 }]:min(1)
      minVal = minVal[1] minPos = minPos[1]
      local notMinPos = -minPos+3;
      
      local d1 = points[{minPos,3}] + points[{notMinPos,3}] 
               - 3*(points[{minPos,2}]-points[{notMinPos,2}])
                     / (points[{minPos,1}]-points[{notMinPos,1}]);
      local d2 = sqrt(d1^2 - points[{minPos,3}]*points[{notMinPos,3}]);

      if isreal(d2) then -- isreal()
         local t = points[{notMinPos,1}] - (points[{notMinPos,1}]
                   - points[{minPos,1}]) * ((points[{notMinPos,3}] + d2 - d1)
                     / (points[{notMinPos,3}] - points[{minPos,3}] + 2*d2))
         
         minPos = min(max(t,points[{minPos,1}]),points[{notMinPos,1}])
      else
         minPos = mean(points[{{},1}])
      end
      return minPos
   end

   -- TODO: get the code below to work!
   error('<optim.polyinterp> extrapolation not implemented yet...')

   -- Compute Bounds of Interpolation Area
   local xmin = points[{{},1}]:min()
   local xmax = points[{{},1}]:max()
   xminBound = xminBound or xmin
   xmaxBound = xmaxBound or xmax

   -- Add constraints on function values
   local A = zeros(nPoints*2,order+1)
   local b = zeros(nPoints*2,1)
   for i = 1,nPoints do
      local constraint = zeros(order+1)
      for j = order,0,-1 do
         constraint[order-j+1] = points[{i,1}]^j
      end
      A[i] = constraint
      b[i] = points[{i,2}]
   end

   -- Add constraints based on derivatives
   for i = 1,nPoints do
      local constraint = zeros(order+1)
      for j = 1,order do
         constraint[j] = (order-j+1)*points[{i,1}]^(order-j)
      end
      A[nPoints+i] = constraint
      b[nPoints+i] = points[{i,3}]
   end

   -- Find interpolating polynomial
   local res = torch.gels(b,A)
   local params = res[{ {1,nPoints*2} }]:squeeze()

   -- Compute Critical Points
   local dParams = zeros(order);
   for i = 1,params:size(1)-1 do
      dParams[i] = params[i]*(order-i+1)
   end

   -- nan/inf?
   local nans = false
   if torch.ne(dParams,dParams):max() > 0 or torch.eq(dParams,math.huge):max() > 0 then
      nans = true
   end
   -- for i = 1,dParams:size(1) do
   --    if dParams[i] ~= dParams[i] or dParams[i] == math.huge then
   --       nans = true
   --       break
   --    end
   -- end
   local cp = torch.cat(Tensor{xminBound,xmaxBound},points[{{},1}])
   if not nans
      local cproots = roots(dParams)
      local cpi = zeros(cp:size(1),2)
      cpi[{ {1,cp:size(1)} , 1 }] = cp
      cp = torch.cat(cpi,cproots)
   end

   -- Test Critical Points
   local fmin = math.huge
   -- Default to Bisection if no critical points valid:
   minPos = (xminBound+xmaxBound)/2
   for i = 1,cp:size(1) do
      local xCP = cp[i]
      if imag(xCP)==0 and xCP >= xminBound and xCP <= xmaxBound then
         local fCP = polyval(params,xCP)
         if imag(fCP)==0 and fCP < fmin then
            minPos = real(xCP)
            fmin = real(fCP)
         end
      end
   end
   return minPos
end
