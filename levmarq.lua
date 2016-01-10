--[[
inputs:
F - vector function at x. (torch.Tensor(m))
J - Jacobian of F at x. (torch.Tensor(m,n)) 
lamda - damping parameter
--]]
calc_lm_step=function(F,J,lamda)
  --F: R^n -> R^m
  m=J:size(1) 
  n=J:size(2) 
  --define M1:=J*J_tr-lambda*I
  --local M1,Id=torch.Tensor(n,n),torch.Tensor(n,n)
  --local d=torch.Tensor(n,1),torch.Tensor(n,1) 
  Id=torch.eye(n)
  M1=torch.mm(J:t(),J)
  M1:add(lamda,Id)
  -- right side:
  g=torch.mv(J:t(),F)
  -- in order to solve linear system with gesv we need to make RS 2-dimensional
  g:resize(g:size(1),1)
  -- solve M1*d=RS
  d=-torch.gesv(g,M1)
  return d,g
  end
  

  --[[
levenberg-marquart optimization algorithm:
-- for a vector function F: R^n -> R^m  the goal is to minimize the function
f(x)=norm(F(x))^2
--at a given point x the levmar-step d is given by:
(J*J_tr+lambda*I)d=-J_tr*F(x)
where J is the jacobian of F
  perform lm step
  input:
  x - current point (should be torch.Tensor(n))
  Feval - vector function F (should return torch.Tensor(m))
  Jeval - Jacobian of F (should return torch.Tensor(m))
  --]]
  function optim.levmar(Feval,Jeval,x,config)
   if config == nil then
      print('no state table, LEVMAR initializing')
   end
  F=Feval(x)
  J=Jeval(x)
  d,g=calc_lm_step(F,J,config.lamda) 
  nrmd=torch.norm(d)
  --print('norm_d='..nrmd)
  fx=torch.norm(Feval(x))^2
  tmp=x+d
  fxnew=torch.norm(Feval(tmp))^2
  if fxnew<fx then
    x:copy(tmp)
    if (config.lamda > config.lamda_min) then
      config.lamda=config.lamda*config.lamda_down
    end
  else
    config.lamda=config.lamda*config.lamda_up
    print('no step in lm_step()')
  end
  --tmp2, tmp3 = model:parameters()
  return fxnew,g
  end
  
