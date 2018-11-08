import sys
import os

################## Args ################
file = './3a/train_language_model.py'
print('Running File:', file[5:])
opt = 'adam' # opt = 'sgd' or 'adam'
LR = 0.001
batch_size = 200
vocab_size = 8000
no_of_hidden_units = 500
no_of_epochs = 75
sequence_length = 50
print('optimizer: %s | LR: %f | batch_size: %d | vocab_size: %d | no_of_hidden_units: %d | no_of_epochs: %d | train_sequence_length: %d' % 
    (opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs, sequence_length))
os.chdir(file[:4])

########################################

os.system('python3.6 '+file[5:]+' %s %f %d %d %d %d %d' %(opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs, sequence_length))
# os.system('python3.6 '+'RNN_test.py'+' %s %f %d %d %d %d' %(opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs))
