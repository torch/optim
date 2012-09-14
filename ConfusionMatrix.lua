----------------------------------------------------------------------
-- A Confusion Matrix class
--
-- Example:
-- conf = optim.ConfusionMatrix( {'cat','dog','person'} )   -- new matrix
-- conf:zero()                                              -- reset matrix
-- for i = 1,N do
--    conf:add( neuralnet:forward(sample), label )          -- accumulate errors
-- end
-- print(conf)                                              -- print matrix
-- image.display(conf:render())                             -- render matrix
--
local ConfusionMatrix = torch.class('optim.ConfusionMatrix')

function ConfusionMatrix:__init(nclasses, classes)
   if type(nclasses) == 'table' then
      classes = nclasses
      nclasses = #classes
   end
   self.mat = torch.FloatTensor(nclasses,nclasses):zero()
   self.valids = torch.FloatTensor(nclasses):zero()
   self.unionvalids = torch.FloatTensor(nclasses):zero()
   self.nclasses = nclasses
   self.totalValid = 0
   self.averageValid = 0
   self.classes = classes or {}
end

function ConfusionMatrix:add(prediction, target)
   if type(prediction) == 'number' then
      -- comparing numbers
      self.mat[target][prediction] = self.mat[target][prediction] + 1
   elseif type(target) == 'number' then
      -- prediction is a vector, then target assumed to be an index
      local prediction_1d = torch.FloatTensor(self.nclasses):copy(prediction)
      local _,prediction = prediction_1d:max(1)
      self.mat[target][prediction[1]] = self.mat[target][prediction[1]] + 1
   else
      -- both prediction and target are vectors
      local prediction_1d = torch.FloatTensor(self.nclasses):copy(prediction)
      local target_1d = torch.FloatTensor(self.nclasses):copy(target)
      local _,prediction = prediction_1d:max(1)
      local _,target = target_1d:max(1)
      self.mat[target[1]][prediction[1]] = self.mat[target[1]][prediction[1]] + 1
   end
end

function ConfusionMatrix:zero()
   self.mat:zero()
   self.valids:zero()
   self.unionvalids:zero()
   self.totalValid = 0
   self.averageValid = 0
end

function ConfusionMatrix:updateValids()
   local total = 0
   for t = 1,self.nclasses do
      self.valids[t] = self.mat[t][t] / self.mat:select(1,t):sum()
      self.unionvalids[t] = self.mat[t][t] / (self.mat:select(1,t):sum()+self.mat:select(2,t):sum()-self.mat[t][t])
      total = total + self.mat[t][t]
   end
   self.totalValid = total / self.mat:sum()
   self.averageValid = 0
   self.averageUnionValid = 0
   local nvalids = 0
   local nunionvalids = 0
   for t = 1,self.nclasses do
      if not sys.isNaN(self.valids[t]) then
         self.averageValid = self.averageValid + self.valids[t]
         nvalids = nvalids + 1
      end
      if not sys.isNaN(self.valids[t]) and not sys.isNaN(self.unionvalids[t]) then
         self.averageUnionValid = self.averageUnionValid + self.unionvalids[t]
         nunionvalids = nunionvalids + 1
      end
   end
   self.averageValid = self.averageValid / nvalids
   self.averageUnionValid = self.averageUnionValid / nunionvalids
end

function ConfusionMatrix:__tostring__()
   self:updateValids()
   local str = 'ConfusionMatrix:\n'
   local nclasses = self.nclasses
   str = str .. '['
   for t = 1,nclasses do
      local pclass = self.valids[t] * 100
      pclass = string.format('%2.3f', pclass)
      if t == 1 then
         str = str .. '['
      else
         str = str .. ' ['
      end
      for p = 1,nclasses do
         str = str .. '' .. string.format('%8d', self.mat[t][p])
      end
      if self.classes and self.classes[1] then
         if t == nclasses then
            str = str .. ']]  ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n'
         else
            str = str .. ']   ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n'
         end
      else
         if t == nclasses then
            str = str .. ']]  ' .. pclass .. '% \n'
         else
            str = str .. ']   ' .. pclass .. '% \n'
         end
      end
   end
   str = str .. ' + average row correct: ' .. (self.averageValid*100) .. '% \n'
   str = str .. ' + average rowUcol correct (VOC measure): ' .. (self.averageUnionValid*100) .. '% \n'
   str = str .. ' + global correct: ' .. (self.totalValid*100) .. '%'
   return str
end

function ConfusionMatrix:render(sortmode, display, block, legendwidth)
   -- args
   local confusion = self.mat
   local classes = self.classes
   local sortmode = sortmode or 'score' -- 'score' or 'occurrence'
   local block = block or 25
   local legendwidth = legendwidth or 200
   local display = display or false

   -- legends
   local legend = {
      ['score'] = 'Confusion matrix [sorted by scores, global accuracy = %0.3f%%, per-class accuracy = %0.3f%%]',
      ['occurrence'] = 'Confusiong matrix [sorted by occurences, accuracy = %0.3f%%, per-class accuracy = %0.3f%%]'
   }

   -- parse matrix / normalize / count scores
   local diag = torch.FloatTensor(#classes)
   local freqs = torch.FloatTensor(#classes)
   local unconf = confusion
   local confusion = confusion:clone()
   local corrects = 0
   local total = 0
   for target = 1,#classes do
      freqs[target] = confusion[target]:sum()
      corrects = corrects + confusion[target][target]
      total = total + freqs[target]
      confusion[target]:div( math.max(confusion[target]:sum(),1) )
      diag[target] = confusion[target][target]
   end

   -- accuracies
   local accuracy = corrects / total * 100
   local perclass = 0
   local total = 0
   for target = 1,#classes do
      if confusion[target]:sum() > 0 then
         perclass = perclass + diag[target]
         total = total + 1
      end
   end
   perclass = perclass / total * 100
   freqs:div(unconf:sum())

   -- sort matrix
   if sortmode == 'score' then
      _,order = sort(diag,1,true)
   elseif sortmode == 'occurrence' then
      _,order = sort(freqs,1,true)
   else
      error('sort mode must be one of: score | occurrence')
   end

   -- render matrix
   local render = zeros(#classes*block, #classes*block)
   for target = 1,#classes do
      for prediction = 1,#classes do
         render[{ { (target-1)*block+1,target*block }, { (prediction-1)*block+1,prediction*block } }] = confusion[order[target]][order[prediction]]
      end
   end

   -- add grid
   for target = 1,#classes do
      render[{ {target*block},{} }] = 0.1
      render[{ {},{target*block} }] = 0.1
   end

   -- create rendering
   require 'qtwidget'
   require 'qttorch'
   local win1 = qtwidget.newimage( (#render)[2]+legendwidth, (#render)[1] )
   image.display{image=render, win=win1}

   -- add legend
   for i in ipairs(classes) do
      -- background cell
      win1:setcolor{r=0,g=0,b=0}
      win1:rectangle((#render)[2],(i-1)*block,legendwidth,block)
      win1:fill()

      -- legend
      win1:setfont(qt.QFont{serif=false, size=fontsize})
      local gscale = diag[order[i]]*0.8+0.2
      win1:setcolor{r=gscale,g=gscale,b=gscale}
      win1:moveto((#render)[2]+10,i*block-block/3)
      win1:show(classes[order[i]])

      -- %
      win1:setfont(qt.QFont{serif=false, size=fontsize})
      local gscale = freqs[order[i]]/freqs:max()*0.9+0.1 --3/4
      win1:setcolor{r=gscale*0.5+0.2,g=gscale*0.5+0.2,b=gscale*0.8+0.2}
      win1:moveto(90+(#render)[2]+10,i*block-block/3)
      win1:show(string.format('[%2.2f%% labels]',math.floor(freqs[order[i]]*10000+0.5)/100))

      for j in ipairs(classes) do
         -- scores
         local score = confusion[order[j]][order[i]]
         local gscale = (1-score)*(score*0.8+0.2)
         win1:setcolor{r=gscale,g=gscale,b=gscale}
         win1:moveto((i-1)*block+block/5,(j-1)*block+block*2/3)
         win1:show(string.format('%02.0f',math.floor(score*100+0.5)))
      end
   end

   -- generate tensor
   local t = win1:image():toTensor()

   -- display
   if display then
      image.display{image=t, legend=string.format(legend[sortmode],accuracy,perclass)}
   end

   -- return rendering
   return t
end
