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
