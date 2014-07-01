require 'optim'


logger_former = optim.Logger('accuracy-former.log')
logger_new = optim.Logger('accuracy-new.log')

logger_new:setNames({'channel 1', 'channel 2', 'channel 3'})

for i = 1, 20 do
   logger_former:add({['channel 1'] = 1 , ['channel 2'] = 0.1 * i, ['channel 3'] = 1 - 0.2 * i})
   logger_new:add({1 , 0.1 * i, 1 - 0.2 * i})
end

logger_former:style({['channel 1'] = '-' , ['channel 2'] = '-', ['channel 3'] = '-'})
logger_new:style{'-', '-', '-'}

logger_former:plot()
logger_new:plot()


