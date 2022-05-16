### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from ImageUtils import CIFAR10Policy, normalise, cutout, normalise_test
from PIL import Image as im
import torch.nn as nn
from tqdm import tqdm
import sys
import subprocess
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

# implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'torchgpipe>'])
# from torchgpipe import GPipe



"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.network = MyNetwork()
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # net = self.network
        
        # self.net = GPipe(nn.Sequential(self.network), balance=[1], chunks=8)
        self.net = self.network
        self.net.to(device)

        #         if device == 'cuda':
        #             net = torch.nn.DataParallel(net)

        self.loss = nn.CrossEntropyLoss().cuda()
        self.learning_rate = self.config.lr
        
        # self.optimizer = torch.optim.Adadelta(self.net.parameters())
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9,
                                         weight_decay=self.config.weight_decay)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=self.config.weight_decay, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.maxepochs)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[40, 80, 120, 160], gamma=0.2)
        # nn.DataParallel(net).parameters()

        ### YOUR CODE HERE

    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        batchsize = self.config.batch_size
        print('### Training... ###')
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            # self.learning_rate = self.learning_rate if epoch > 2 else self.learning_rate * 0.1
            # if epoch * num_batches > 48e3:
            #     self.learning_rate *= 1e-2
            # elif epoch * num_batches > 32e3:
            #     self.learning_rate *= 1e-1
            num_batches = int(num_samples / batchsize)
            ### YOUR CODE HERE

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                start_index = i * batchsize
                end_index = start_index + batchsize
                if end_index > curr_x_train.shape[0]:  # shouldn't happen bc of the way num_batches is calculated but still.
                    continue
                policy = CIFAR10Policy()
                # x_batch = np.array([np.transpose(normalise(cutout(np.array(policy(im.fromarray(np.transpose(curr_x_train[ix].reshape(3,32,32),[1,2,0])))))), [2, 0, 1]) for ix in range(start_index, end_index)])
                x_batch = torch.stack([normalise_test(cutout(np.array(policy(im.fromarray(np.transpose(curr_x_train[ix].reshape(3,32,32),[1,2,0])))))) for ix in range(start_index, end_index)])

                #                 x_batch = [parse_record(x, True) for x in curr_x_train[start_index:end_index]]
                # x_batch = [parse_record(curr_x_train[ix], True) for ix in range(start_index, end_index)]
                y_batch = np.array(curr_y_train[start_index:end_index])

                

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
              
                # inputs, targets = torch.tensor(x_batch).float().to(device), torch.tensor(y_batch).long().to(device)
                inputs, targets = x_batch.float().to(device), torch.from_numpy(y_batch).long().to(device)
                outputs = self.net(inputs)
                loss = self.loss(outputs, targets)

                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            self.scheduler.step()
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)

    def evaluate(self, x, y, checkpoint_num_list):
        self.net.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.datadir, self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)

            preds = []
        
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                # inputs = torch.tensor(parse_record(x[i])).float().to(
                #     device)  
                inputs = parse_record(x[i]).float().to(
                    device)
                
                inputs = inputs.view(1, 3, 32, 32)
                
                outputs = self.net(inputs)
                y_i = torch.tensor(y[i])
                predicted = int(torch.argmax(torch.exp(outputs.data), 1))
                # total += y.size(0)
                
                # prediction = int(torch.max(outputs.data, 1)[1])
                # prediction = int(torch.argmax(outputs.data, 1))
                preds.append(predicted)
                

            correct = torch.tensor(preds).eq(torch.tensor(y)).sum().item()
            # preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(100.*correct/len(y)))#torch.sum(preds == y) / y.shape[0]))

    def predict_prob(self, x, checkpoint_num_list):
        self.net.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.datadir, self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)

            preds = []
        
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                # self.network = self.network.to("cuda")
                device = 'cuda'
                # inputs = torch.from_numpy(normalise_test(x[i])).float().to(
                #     device) 
                inputs = normalise_test(x[i]).float().to(
                    device)  
                inputs = inputs.view(1, 3, 32, 32)
                
                outputs = torch.exp(self.net(inputs))
                # outputs = int(torch.argmax(torch.exp(self.net(inputs).data),1))
                
                # total += y.size(0)
                
                # prediction = int(torch.max(outputs.data, 1)[1])
                # prediction = int(torch.argmax(outputs.data, 1))
                preds.append(outputs.cpu().detach().numpy()[0])
                # preds.append(outputs)
        return np.array(preds)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.datadir, self.config.modeldir, 'model-%d.ckpt' % (epoch))
        os.makedirs(self.config.datadir + "/" + self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))