require 'torch'
require 'optim'

n_feature = 3
classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

print'ConfusionMatrix:__init() test'
cm = optim.ConfusionMatrix(#classes, classes)

target = 3
prediction = torch.randn(#classes)

print'ConfusionMatrix:add() test'
cm:add(prediction, target)
cm:add(prediction, torch.randn(#classes))

batch_size = 8

targets = torch.randperm(batch_size)
predictions = torch.randn(batch_size, #classes)

print'ConfusionMatrix:batchAdd() test'
cm:batchAdd(predictions, targets)
assert(cm.mat:sum() == batch_size + 2, 'missing examples')

print'ConfusionMatrix:updateValids() test'
cm:updateValids()

print'ConfusionMatrix:__tostring__() test'
print(cm)

target = 0
cm:add(prediction, target)
assert(cm.mat:sum() == batch_size + 2, 'too many examples')

-- FAR/FRR testing on identify matrix. FRR/FAR should be zero for identity.
cm.mat = torch.eye(#classes, #classes)
classFrrs, classFars, frrs, fars = cm:farFrr()
assert(classFrrs:sum() + classFars:sum() == 0, "Incorrect values")
