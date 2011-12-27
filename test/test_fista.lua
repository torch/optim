
require 'unsup'
require 'lab'
require 'plot'

-- plot.setgnuplotexe('/usr/bin/gnuplot44')
-- plot.setgnuplotterminal('x11')

function gettableval(tt,v)
   local x = torch.Tensor(#tt)
   for i=1,#tt do x[i] = tt[i][v] end
   return x
end
function doplots(v)
   v = v or 'F'
   local fistaf = torch.DiskFile('fista2.bin'):binary()
   local istaf = torch.DiskFile('ista2.bin'):binary()
   
   local hfista = fistaf:readObject()
   fistaf:close()
   local hista = istaf:readObject()
   istaf:close()

   plot.figure()
   plot.plot({'fista ' .. v,gettableval(hfista,v)},{'ista ' .. v, gettableval(hista,v)})
end

seed = seed or 123
if dofista == nil then
   dofista = true
else
   dofista = not dofista
end

random.manualSeed(seed)
math.randomseed(seed)
nc = 3
ni = 30
no = 100
x = torch.Tensor(ni):zero()

--- I am keeping these just to make sure random init stays same
fista = unsup.LinearFistaL1(ni,no,0.1)
fista = nil

fistaparams = {}
fistaparams.doFistaUpdate = dofista
fistaparams.maxline = 10
fistaparams.maxiter = 200
fistaparams.verbose = true

D=lab.randn(ni,no)
for i=1,D:size(2) do
   D:select(2,i):div(D:select(2,i):std()+1e-12)
end

mixi = torch.Tensor(nc)
mixj = torch.Tensor(nc)
for i=1,nc do
   local ii = math.random(1,no)
   local cc = random.uniform(0,1/nc)
   mixi[i] = ii;
   mixj[i] = cc;
   print(ii,cc)
   x:add(cc, D:select(2,ii))
end

fista = optim.FistaL1(D,fistaparams)
code,h = fista.run(x,0.1)

--fista.reconstruction:addmv(0,1,D,code)
rec = fista.reconstruction
--code,rec,h = fista:forward(x);

plot.figure(1)
plot.plot({'data',mixi,mixj,'+'},{'code',lab.linspace(1,no,no),code,'+'})
plot.title('Fista = ' .. tostring(fistaparams.doFistaUpdate))

plot.figure(2)
plot.plot({'input',lab.linspace(1,ni,ni),x,'+-'},{'reconstruction',lab.linspace(1,ni,ni),rec,'+-'});
plot.title('Reconstruction Error : ' ..  x:dist(rec) .. ' ' .. 'Fista = ' .. tostring(fistaparams.doFistaUpdate))
--w2:axis(0,ni+1,-1,1)

if dofista then
   print('Running FISTA')
   fname = 'fista2.bin'
else
   print('Running ISTA')
   fname = 'ista2.bin'
end
ff = torch.DiskFile(fname,'w'):binary()
ff:writeObject(h)
ff:close()

