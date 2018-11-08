import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import time
import os
import sys
import io

from RNN_model import RNN_model

print('RNN Test 2b')
################## Args ################
if len(sys.argv) == (1 + 6):
    opt = str(sys.argv[1]) # opt = 'sgd' or 'adam'
    LR = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    vocab_size = int(sys.argv[4])
    no_of_hidden_units = int(sys.argv[5])
    no_of_epochs = int(sys.argv[6])
    print('optimizer: %s | LR: %f | batch_size: %d | vocab_size: %d | no_of_hidden_units: %d | no_of_epochs: %d' % 
        (opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs))
else:
    raise Exception('Incorrect number of arguments. Program exit!')

########################################

rnn = torch.load('rnn_GloVe.model')

is_cuda = torch.cuda.is_available()
print('Cuda is', is_cuda)


glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
# vocab_size = 100000

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

if is_cuda:
    rnn.cuda()
    rnn = nn.DataParallel(rnn, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    # rnn.cuda()


## Define the Model with desired learning rate ##
# opt = 'sgd'
# LR = 0.01
#opt = 'adam'
#LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(rnn.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(rnn.parameters(), lr=LR, momentum=0.9)



#batch_size = 200
#no_of_epochs = 10
L_Y_train = len(y_train)
L_Y_test = len(y_test)

train_loss = []
train_accu = []
test_accu = []


for epoch in range(2):#no_of_epochs):
    # ## test
    print('Testing on Epoch', (epoch+1))
    rnn.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = (epoch+1)*50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input] # Extract glove_embeddings
        y_input = y_train[I_permutation[i:i+batch_size]]

        if is_cuda:
            data = Variable(torch.FloatTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()
        else:
            data = Variable(torch.FloatTensor(x_input))
            target = Variable(torch.FloatTensor(y_input))

        with torch.no_grad():
            loss, pred = rnn(data,target, train=False)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("sequence_length:  %d" %(sequence_length), "| Test accuracy %.2f" % (epoch_acc*100.0), " | Loss: %.4f" % epoch_loss, " | Elapsed time:", time_elapsed)

data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('test2b.npy',data)