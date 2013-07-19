----------------------------------------------------------------------
-- Logger: a simple class to log symbols during training,
--         and automate plot generation
--
-- Example:
-- logger = optim.Logger('somefile.log')    -- file to save stuff
-- 
-- for i = 1,N do                           -- log some symbols during
--    train_error = ...                     -- training/testing
--    test_error = ...
--    logger:add{['training error'] = train_error, 
--               ['test error'] = test_error}
-- end
-- 
-- logger:style{['training error'] = '-',   -- define styles for plots
--              ['test error'] = '-'}
-- logger:plot()                            -- and plot
--
local Logger = torch.class('optim.Logger')

function Logger:__init(filename, timestamp)
   if filename then
      self.name = filename
      os.execute('mkdir -p "' .. sys.dirname(filename) .. '"')
      if timestamp then
         -- append timestamp to create unique log file
         filename = filename .. '-'..os.date("%Y_%m_%d_%X")
      end
      self.file = io.open(filename,'w')
      self.epsfile = self.name .. '.eps'
   else
      self.file = io.stdout
      self.name = 'stdout'
      print('<Logger> warning: no path provided, logging to std out') 
   end
   self.empty = true
   self.symbols = {}
   self.styles = {}
   self.figure = nil
end

function Logger:add(symbols)
   -- (1) first time ? print symbols' names on first row
   if self.empty then
      self.empty = false
      self.nsymbols = #symbols
      for k,val in pairs(symbols) do
         self.file:write(k .. '\t')
         self.symbols[k] = {}
         self.styles[k] = {'+'}
      end
      self.file:write('\n')
   end
   -- (2) print all symbols on one row
   for k,val in pairs(symbols) do
      if type(val) == 'number' then
         self.file:write(string.format('%11.4e',val) .. '\t')
      elseif type(val) == 'string' then
         self.file:write(val .. '\t')
      else
         xlua.error('can only log numbers and strings', 'Logger')
      end
   end
   self.file:write('\n')
   self.file:flush()
   -- (3) save symbols in internal table
   for k,val in pairs(symbols) do
      table.insert(self.symbols[k], val)
   end
end

function Logger:style(symbols)
   for name,style in pairs(symbols) do
      if type(style) == 'string' then
         self.styles[name] = {style}
      elseif type(style) == 'table' then
         self.styles[name] = style
      else
         xlua.error('style should be a string or a table of strings','Logger')
      end
   end
end

function Logger:plot(...)
   if not xlua.require('gnuplot') then
      if not self.warned then 
         print('<Logger> warning: cannot plot with this version of Torch') 
         self.warned = true
      end
      return
   end
   local plotit = false
   local plots = {}
   local plotsymbol = 
      function(name,list)
         if #list > 1 then
            local nelts = #list
            local plot_y = torch.Tensor(nelts)
            for i = 1,nelts do
               plot_y[i] = list[i]
            end
            for _,style in ipairs(self.styles[name]) do
               table.insert(plots, {name, plot_y, style})
            end
            plotit = true
         end
      end
   local args = {...}
   if not args[1] then -- plot all symbols
      for name,list in pairs(self.symbols) do
         plotsymbol(name,list)
      end
   else -- plot given symbols
      for i,name in ipairs(args) do
         plotsymbol(name,self.symbols[name])
      end
   end
   if plotit then
      self.figure = gnuplot.figure(self.figure)
      gnuplot.plot(plots)
      gnuplot.title('<Logger::' .. self.name .. '>')
      if self.epsfile then
         os.execute('rm -f "' .. self.epsfile .. '"')
         local epsfig = gnuplot.epsfigure(self.epsfile)
         gnuplot.plot(plots)
         gnuplot.title('<Logger::' .. self.name .. '>')
         gnuplot.plotflush()
         gnupot.close(epsfig)
      end
   end
end
