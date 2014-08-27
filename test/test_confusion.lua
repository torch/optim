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

batch_size = 8

targets = torch.randperm(batch_size)
predictions = torch.randn(batch_size, #classes)

print'ConfusionMatrix:batchAdd() test'
cm:batchAdd(predictions, targets)
assert(cm.mat:sum() == batch_size + 1, 'missing examples')

print'ConfusionMatrix:updateValids() test'
cm:updateValids()

print'ConfusionMatrix:__tostring__() test'
print(cm)



--[[
The following code test: MCC, F1, sensitivity, specificity, positive predictive
value, negative predictive value, false positive rate, false discovery rate and
class accuracy.

See http://en.wikipedia.org/wiki/Sensitivity_and_specificity for definitions

]]
print("------------------------------")
print('Testing statistics functions')

function checkequality(t1,t2,prec,pr)
     if pr then
          print(t1)
          print(t2)
     end 
     local prec = prec or -4
     assert(torch.numel(t1)==torch.numel(t1))
     
     local res = torch.add(t1,-t2):abs()
     res = torch.le(res, math.pow(10,prec))
     res = res:sum()
     local ret
     if torch.numel(t1) == res then
          ret = true
     else
          ret = false
     end

     return ret

end

function numeq(a,b)
	return math.abs(a-b) < math.pow(10,-3)
end

 classes = {'A', 'B','C'}
 c = optim.ConfusionMatrix(classes)



 -- ADD some examples
 AC = 2; BC = 2; CC  = 5; CA = 6; AA = 7

 c.mat[{3,1}] = AC   					--    AC = predict A true C 
 c.mat[{3,2}] = BC  					--    BC = predict B true C 
 c.mat[{3,3}] = CC   					--    CC = predict C true C 

  -- ADD 7 FP for class C
  c.mat[{1,3}] = CA  					--    CA = Predict C true A


   -- ADD 7 TP for class A 
   c.mat[{1,1}] = AA                    --	  AA = predict A true A 

tp, tn, fp, fn=c:getConfusion()

-- Check statistics  / check getErrors function
fp_test = torch.Tensor({AC,BC,CA}):resize(1,3):float()
fn_test = torch.Tensor({CA,0,AC+BC}):resize(1,3):float()
tp_test = torch.Tensor({AA,0,CC}):resize(1,3):float()
tn_test = torch.Tensor({BC+CC,AC+CC+CA+AA,AA}):resize(1,3):float()


checkequality(tp,tp_test,-10)
checkequality(tn,tn_test,-10)
checkequality(fp,fp_test,-10)
checkequality(fn,fn_test,-10)

-- MCC
mcc = c:matthewsCorrelation()
test_num = tp[{1,3}] *tn[{1,3}] - fp[{1,3}] * fn[{1,3}] 
test_denom = math.sqrt(
		     (tp[{1,3}] + fp[{1,3}])*(tp[{1,3}] + fn[{1,3}]) *
			 (tn[{1,3}] + fp[{1,3}])*(tn[{1,3}] + fn[{1,3}])
			 )
C_mcc  = test_num / test_denom
assert(numeq(C_mcc,mcc[{1,3}]))

-- Sensitivity
A_sens = tp[{1,1}] /  (tp[{1,1}] + fn[{1,1}] )
B_sens = 0 -- Division by zero 
C_sens = tp[{1,3}] /  (tp[{1,3}] + fn[{1,3}] )
sens_test = torch.Tensor({A_sens,B_sens,C_sens}):resize(1,3):float()
checkequality(sens_test,c:sensitivity(),-10)

-- Specificty
spec = c:specificity()
C_spec = tn[{1,3}] / (fp[{1,3}] + tn[{1,3}] )
assert(numeq(C_spec,spec[{1,3}]))


-- PPV
ppv = c:positivePredictiveValue()
A_ppv = tp[{1,1}] / ( tp[{1,1}]+fp[{1,1}] )
assert(numeq(A_ppv,ppv[{1,1}]))

-- NPV
npv = c:negativePredictiveValue()
A_npv = tn[{1,1}] / ( tn[{1,1}]+fn[{1,1}] )
assert(numeq(A_npv,npv[{1,1}]))

--FPR
fpr = c:falsePositiveRate()
C_fpr = fp[{1,3}] / ( fp[{1,3}] + tn[{1,3}])
B_fpr = fp[{1,2}] / ( fp[{1,2}] + tn[{1,2}])
assert(numeq(C_fpr,fpr[{1,3}]))
assert(numeq(B_fpr,fpr[{1,2}]))

-- FDR
fdr = c:falseDiscoveryRate()
B_fdr = fp[{1,2}] / (tp[{1,2}] + fp[{1,2}] )
assert(numeq(B_fdr,fdr[{1,2}]))


-- Class ACC
acc = c:classAccuracy()
A_acc = (tp[{1,1}] + tn[{1,1}]) / (tp[{1,1}] + tn[{1,1}] + fp[{1,1}] + fn[{1,1}])
B_acc = (tp[{1,2}] + tn[{1,2}]) / (tp[{1,2}] + tn[{1,2}] + fp[{1,2}] + fn[{1,2}])
C_acc = (tp[{1,3}] + tn[{1,3}]) / (tp[{1,3}] + tn[{1,3}] + fp[{1,3}] + fn[{1,3}])
assert(numeq(A_acc,acc[{1,1}]))
assert(numeq(B_acc,acc[{1,2}]))
assert(numeq(C_acc,acc[{1,3}]))

-- F1
f1 = c:F1()
C_f1 = tp[{1,3}]*2 / (tp[{1,3}]*2 + fp[{1,3}]+fn[{1,3}])
assert(numeq(C_f1,f1[{1,3}]))


-- Further tests
tn = torch.Tensor({825}):resize(1,1):float()
tp = torch.Tensor({75}):resize(1,1):float()
fn = torch.Tensor({25}):resize(1,1):float()
fp = torch.Tensor({75}):resize(1,1):float()

c1 = optim.ConfusionMatrix({'A','B'})
c1.mat[{1,1}] = tn[{1,1}]
c1.mat[{2,2}] = tp[{1,1}]
c1.mat[{2,1}] = fn[{1,1}]
c1.mat[{1,2}] = fp[{1,1}]

mcc = -- 
assert( numeq(0.560, c1:matthewsCorrelation()[{1,1}], -3) )
assert( numeq(0.75, c1:sensitivity()[{1,2}], -3) )
assert( numeq(0.6, c1:F1()[{1,2}], -3) )

print('OK')
print("------------------------------")







