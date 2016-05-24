--[[ An implementation of `DE` (Differential Evolution), 

ARGS:

-`opfunc` : a function that takes a single input (X), the point of 
               evaluation, and returns f(X) and df/dX. Note that df/dX is not used 
-`x` : 		the initial point
-`state.popsize`: 			population size. If this is left empty, 10*d will be used
-`state.scaleFactor`: 		float, usually between 0.4 and 1 
-`state.crossoverRate`:		float, usually between 0.1 and 0.9
-`state.maxEval`:			int, maximal number of function evaluations

RETURN:
- `x*` : the new `x` vector, at the optimal point
- `f`  : a table of all function values: 
       `f[1]` is the value of the function before any optimization and
       `f[#f]` is the final fully optimized value, at `x*`
--]]

require 'torch'

function optim.de(opfunc, x, config, state)
	-- process input parameters
	local config = config or {}
	local state = state
	local popsize = config.popsize			  	-- population size
	local scaleFactor = config.scaleFactor	 	-- scale factor
	local crossoverRate = config.crossoverRate	-- crossover rate
	local maxFEs = tonumber(config.maxFEs)		-- maximal number of function evaluations
	local maxRegion = config.maxRegion			-- upper bound of search region
	local minRegion = config.minRegion			-- lower bound of search region
	local xmean = x:clone():view(-1) 			-- distribution mean, a flattened copy
	local D = xmean:size(1)  					-- number of objective variables/problem dimension


	if config.popsize == nil then
    	popsize = 10*D
   	end 
   	if config.maxRegion == nil then
   		maxRegion = 30
   	end
   	if config.minRegion == nil then
   		minRegion = -30
   	end

   	-- Initialize population 

   	local pop = torch.Tensor(popsize,D)
   	local children = torch.Tensor(popsize,D)
   	local fitness = torch.Tensor(popsize)
   	local children_fitness = torch.Tensor(popsize)
   	local fes = 0	-- number of function evaluations
   	local best_fitness

	-- Initialize population and evaluate the its fitness value
   	local gen = torch.Generator()
   	torch.manualSeed(gen, 1)
   	for i=1,popsize do
   		for j=1,D do
   			pop[i][j] = torch.uniform(gen, minRegion, maxRegion)
   		end
   		fitness[i] = opfunc(pop[i])
   		fes = fes + 1
   	end
   	
   	-- Find the best solution
   	best_fitness = fitness[1]
   	for i=2,popsize do
   		if best_fitness > fitness[i] then
   			best_fitness = fitness[i]
   		end
   	end

   	-- Main loop
   	while fes < maxFEs do
   		local  r1, r2, r3
   		for i=1,popsize do
   			repeat
   				r1 = torch.uniform(gen, 1, popsize)
   			until(r1 ~= i)
   			repeat
   				r2 = torch.uniform(gen, 1, popsize)
   			until(r2 ~= r1 and r2 ~= i)
   			repeat
   				r3 = torch.uniform(gen, 1, popsize)
   			until(r3 ~= r2 and r3 ~= r1 and r3 ~= i)
   			local jrand = torch.uniform(gen, 1, D)
   			for j=1,D do
   				if torch.uniform(gen, 0, 1) < crossoverRate or i == jrand then
   					children[i][j] = pop[r1][j] + scaleFactor * (pop[r2][j] - pop[r3][j])
   				else
   					children[i][j] = pop[i][j]
   				end
   			end
   			children_fitness[i] = opfunc(children[i])
   			fes = fes + 1
   		end

   		for i=1,popsize do
   			if children_fitness[i] <= fitness[i] then
   				for j=1,D do
   					pop[i][j] = children[i][j]
   				end
   				fitness[i] = children_fitness[i]
   				if fitness[i] < best_fitness then
   					best_fitness = fitness[i]
   				end
   			end
   		end
   	end
   	return best_fitness
end
