import sys
import os

################## Args ################
file = './2a/RNN_sentiment_analysis.py'
print('Running File:', file[5:])
opt = 'adam' # opt = 'sgd' or 'adam'
LR = 0.001
batch_size = 200
vocab_size = 800
no_of_hidden_units = 500
no_of_epochs = 6
print('optimizer: %s | LR: %f | batch_size: %d | vocab_size: %d | no_of_hidden_units: %d | no_of_epochs: %d' % 
    (opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs))
os.chdir(file[:4])

########################################

os.system('python3.6 '+file[5:]+' %s %f %d %d %d %d' %(opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs))