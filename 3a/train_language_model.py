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

from RNN_language_model import RNN_language_model



################## Args ################
if len(sys.argv) == (1 + 7):
    opt = str(sys.argv[1]) # opt = 'sgd' or 'adam'
    LR = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    vocab_size = int(sys.argv[4])
    no_of_hidden_units = int(sys.argv[5])
    no_of_epochs = int(sys.argv[6])
    sequence_length = int(sys.argv[7])
    print('optimizer: %s | LR: %f | batch_size: %d | vocab_size: %d | no_of_hidden_units: %d | no_of_epochs: %d | sequence_length_train: %d' % 
        (opt, LR, batch_size, vocab_size, no_of_hidden_units, no_of_epochs, sequence_length))
else:
    raise Exception('Incorrect number of arguments. Program exit!')

########################################

print('train_language model 3a')
is_cuda = torch.cuda.is_available()
print('Cuda is', is_cuda)


imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
# vocab_size = 8000

# The last three lines of code just grab the first 25,000 sequences
#  and creates a vector for the labels (the first 125000 are labeled 1 
# for positive and the last 12500 are labeled 0 for negative).
x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
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

# do the same for test set
x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1


# no_of_hidden_units = 500
vocab_size += 1


model = RNN_language_model(vocab_size, no_of_hidden_units)
if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    # rnn.cuda()


## Define the Model with desired learning rate ##
# opt = 'sgd'
# LR = 0.01
# opt = 'adam'
# LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)



# batch_size = 200
# no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

train_loss = []
train_accu = []
test_accu = []

print('begin training...')
for epoch in range(no_of_epochs): # 75

    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR/10.0

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl<sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]

        if is_cuda:
        	x_input = Variable(torch.LongTensor(x_input)).cuda()
        else:
            x_input = Variable(torch.LongTensor(x_input))


        optimizer.zero_grad()
        loss, pred = model(x_input)
        loss.backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(),2.0)

        optimizer.step()   # update gradients
        
        values,prediction = torch.max(pred,1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
        epoch_acc += accuracy
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
        
        if (i+batch_size) % 1000 == 0 and epoch==0:
           print(i+batch_size, accuracy/batch_size, loss.data.item(), norm, "%.4f" % float(time.time()-time1))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)


    print('Train | Epoch:',epoch, "| Epoch_Acc %.2f" % (epoch_acc*100.0), "| Loss %.4f" % epoch_loss, "| Time %.4f" % float(time.time()-time1))

    ## test
    if((epoch+1)%1==0):
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            sequence_length = 100
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl<sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            if is_cuda:
            	x_input = Variable(torch.LongTensor(x_input)).cuda()
            else:
            	x_input = Variable(torch.LongTensor(x_input))



            with torch.no_grad():
                pred = model(x_input,train=False)
            
            values,prediction = torch.max(pred,1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
            epoch_acc += accuracy
            epoch_loss += loss.data.item()
            epoch_counter += batch_size
            #train_accu.append(accuracy)
            if (i+batch_size) % 1000 == 0 and epoch==0:
               print(i+batch_size, accuracy/batch_size)
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("Test:  ", "%.2f" % (epoch_acc*100.0), " | Loss: %.4f" % epoch_loss, " | Time: %.4f" %float(time.time()-time1))

    if is_cuda:
    	torch.cuda.empty_cache()

    if(((epoch+1)%2)==0):
        torch.save(model,'temp.model')
        torch.save(optimizer,'temp.state')
        data = [train_loss,train_accu,test_accu]
        data = np.asarray(data)
        np.save('data.npy',data)
torch.save(model,'language.model')


