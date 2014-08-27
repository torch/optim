--[[ A Confusion Matrix class

Example:

    conf = optim.ConfusionMatrix( {'cat','dog','person'} )   -- new matrix
    conf:zero()                                              -- reset matrix
    for i = 1,N do
        conf:add( neuralnet:forward(sample), label )         -- accumulate errors
    end
    print(conf)                                              -- print matrix
    image.display(conf:render())                             -- render matrix
]]
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
      self.prediction_1d = self.prediction_1d or torch.FloatTensor(self.nclasses)
      self.prediction_1d:copy(prediction)
      local _,prediction = self.prediction_1d:max(1)
      self.mat[target][prediction[1]] = self.mat[target][prediction[1]] + 1
   else
      -- both prediction and target are vectors
      self.prediction_1d = self.prediction_1d or torch.FloatTensor(self.nclasses)
      self.prediction_1d:copy(prediction)
      self.target_1d = self.target_1d or torch.FloatTensor(self.nclasses)
      self.target_1d:copy(target)
      local _,prediction = self.prediction_1d:max(1)
      local _,target = self.target_1d:max(1)
      self.mat[target[1]][prediction[1]] = self.mat[target[1]][prediction[1]] + 1
   end
end

function ConfusionMatrix:batchAdd(predictions, targets)
   local preds, targs, __
   if predictions:dim() == 1 then
      -- predictions is a vector of classes
      preds = predictions
   elseif predictions:dim() == 2 then
      -- prediction is a matrix of class likelihoods
      if predictions:size(2) == 1 then
         -- or prediction just needs flattening
         preds = predictions:select(2,1)
      else
         __,preds = predictions:max(2)
         preds:resize(preds:size(1))
      end
   else
      error("predictions has invalid number of dimensions")
   end
      
   if targets:dim() == 1 then
      -- targets is a vector of classes
      targs = targets
   elseif targets:dim() == 2 then
      -- targets is a matrix of one-hot rows
      if targets:size(2) == 1 then
         -- or targets just needs flattening
         targs = targets:select(2,1)
      else
         __,targs = targets:max(2)
         targs:resize(targs:size(1))
      end
   else
      error("targets has invalid number of dimensions")
   end
   --loop over each pair of indices
   for i = 1,preds:size(1) do
      self.mat[targs[i]][preds[i]] = self.mat[targs[i]][preds[i]] + 1
   end
end

function ConfusionMatrix:zero()
   self.mat:zero()
   self.valids:zero()
   self.unionvalids:zero()
   self.totalValid = 0
   self.averageValid = 0
end

local function isNaN(number)
  return number ~= number
end

local function remNaN(x,self)
      for i = 1, self.nclasses  do
      if isNaN(x[{1,i}]) then
         x[{1,i}] = 0
      end
   end
   return x
end


local function getErrors(self) 
   local tp, fn, fp, tn
   tp  = torch.diag(self.mat):resize(1,self.nclasses )
   fn = (torch.sum(self.mat,2)-torch.diag(self.mat)):t()
   fp = torch.sum(self.mat,1)-torch.diag(self.mat)
   tn  = torch.Tensor(1,self.nclasses):fill(torch.sum(self.mat)):typeAs(tp) - tp - fn - fp

    return tp, tn, fp, fn
   end


function ConfusionMatrix:getConfusion()
   return getErrors(self)
end

function ConfusionMatrix:matthewsCorrelation()
   local mcc,numerator, denominator
   tp, tn, fp, fn = getErrors(self) 
   numerator = torch.cmul(tp,tn) - torch.cmul(fp,fn)
   denominator = torch.sqrt((tp+fp):cmul(tp+fn):cmul(tn+fp):cmul(tn+fn))
   mcc = torch.cdiv(numerator,denominator)
   mcc = remNaN(mcc,self)
   return mcc
end

function ConfusionMatrix:sensitivity()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(tp, tp + fn )
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:specificity()
   tp, tn, fp, fn = getErrors(self)
   res =  torch.cdiv(tn, tn + fp)  -- TN / (TN + FP)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)

end

function ConfusionMatrix:positivePredictiveValue()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(tp, tp + fp )  -- TP / (TP + FP)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:negativePredictiveValue()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(tn, tn + fn )  -- TN / (TN + FN)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:falsePositiveRate()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(fp, fp + tn)  -- FP / (FP + TN)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:falseDiscoveryRate()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(fp, tp + fp)  -- FP / (TP + FP)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:classAccuracy()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(tp + tn, tp + tn + fp + fn)  -- (TP + FN) / (TN + TP + FN + FP)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
end

function ConfusionMatrix:F1()
   tp, tn, fp, fn = getErrors(self)
   res = torch.cdiv(tp * 2, tp * 2 + fp + fn)  -- (2*TP)/(TP*2+fp+fn)
   res = remNaN(res,self)
   return res -- TP / (TP + FN)
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
      if not isNaN(self.valids[t]) then
         self.averageValid = self.averageValid + self.valids[t]
         nvalids = nvalids + 1
      end
      if not isNaN(self.valids[t]) and not isNaN(self.unionvalids[t]) then
         self.averageUnionValid = self.averageUnionValid + self.unionvalids[t]
         nunionvalids = nunionvalids + 1
      end
   end
   self.averageValid = self.averageValid / nvalids
   self.averageUnionValid = self.averageUnionValid / nunionvalids
end

function ConfusionMatrix:__tostring__()
   self:updateValids()
   local str = {'ConfusionMatrix:\n'}
   local nclasses = self.nclasses
   table.insert(str, '[')
   for t = 1,nclasses do
      local pclass = self.valids[t] * 100
      pclass = string.format('%2.3f', pclass)
      if t == 1 then
         table.insert(str, '[')
      else
         table.insert(str, ' [')
      end
      for p = 1,nclasses do
         table.insert(str, string.format('%8d', self.mat[t][p]))
      end
      if self.classes and self.classes[1] then
         if t == nclasses then
            table.insert(str, ']]  ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n')
         else
            table.insert(str, ']   ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n')
         end
      else
         if t == nclasses then
            table.insert(str, ']]  ' .. pclass .. '% \n')
         else
            table.insert(str, ']   ' .. pclass .. '% \n')
         end
      end
   end
   table.insert(str, ' + average row correct: ' .. (self.averageValid*100) .. '% \n')
   table.insert(str, ' + average rowUcol correct (VOC measure): ' .. (self.averageUnionValid*100) .. '% \n')
   table.insert(str, ' + global correct: ' .. (self.totalValid*100) .. '%')
   return table.concat(str)
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
      _,order = torch.sort(diag,1,true)
   elseif sortmode == 'occurrence' then
      _,order = torch.sort(freqs,1,true)
   else
      error('sort mode must be one of: score | occurrence')
   end

   -- render matrix
   local render = torch.zeros(#classes*block, #classes*block)
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
   require 'image'
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
      
      -- %
      win1:setfont(qt.QFont{serif=false, size=fontsize})
      local gscale = freqs[order[i]]/freqs:max()*0.9+0.1 --3/4
      win1:setcolor{r=gscale*0.5+0.2,g=gscale*0.5+0.2,b=gscale*0.8+0.2}
      win1:moveto((#render)[2]+10,i*block-block/3)
      win1:show(string.format('[%2.2f%% labels]',math.floor(freqs[order[i]]*10000+0.5)/100))

      -- legend
      win1:setfont(qt.QFont{serif=false, size=fontsize})
      local gscale = diag[order[i]]*0.8+0.2
      win1:setcolor{r=gscale,g=gscale,b=gscale}
      win1:moveto(120+(#render)[2]+10,i*block-block/3)
      win1:show(classes[order[i]])

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
