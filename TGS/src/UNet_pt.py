from unet.unet_model import UNet

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Function, Variable

from sklearn.model_selection import train_test_split
#import utils
import data_utils
import argparse
import config

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

class UNet_pt:
    ''''''
    def __init__(self, input_channel= 3, output_channel= 1):
        ''''''
        self.network= UNet(input_channel, output_channel).cuda()

    def __dice_coeff(self, predict, truth):
        """Dice coeff for batches"""
        if predict.is_cuda:
            s = torch.Tensor(1).cuda().zero_()
        else:
            s = torch.Tensor(1).zero_()

        for i, c in enumerate(zip(predict, truth)):
            s = s + DiceCoeff().forward(c[0], c[1])

        return s / (i + 1)

    def __eval_net(self, net, X, Y):
        """Evaluation without the densecrf with the dice coefficient"""
        tot = 0
        for i, b in enumerate(zip(X, Y)):
            img = b[0]
            true_mask = b[1]

            img = torch.Tensor(img).unsqueeze(0)
            true_mask = torch.Tensor(true_mask).unsqueeze(0)

            img = img.cuda()
            true_mask = true_mask.cuda()

            mask_pred = net(img)[0]
            mask_pred = torch.Tensor.double(((F.sigmoid(mask_pred) > 0.5)))

            tot += self.__dice_coeff(mask_pred, true_mask).item()

        return tot / i

    def __batch(self, iterable, batch_size):
        """Yields lists by batch"""
        b = []
        for i, t in enumerate(iterable):
            b.append(t)
            if (i + 1) % batch_size == 0:
                yield b
                b = []

        if len(b) > 0:
            yield b

    def fit(self, X_train, Y_train, X_valid, Y_valid, batch_size, epochs, model_ckpt, learning_rate= 0.01):

        optimizer= optim.Adam(self.network.parameters(), lr= learning_rate, weight_decay= 0.0005) # adam optimizer
        criterion = nn.BCELoss() # binary cross entropy

        train_size = len(X_train) # ndarray
        train_valid = zip(X_train, Y_train, X_valid, Y_valid) # ndarray

        for e in range(epochs):

            epoch_loss = .0

            net = self.network.train(mode= True)

            for batch_no, (X_train_b, Y_train_b, X_valid_b, Y_valid_b) in enumerate(self.__batch(train_valid, batch_size)):

                imgs = torch.Tensor(X_train_b).cuda()
                true_masks = torch.Tensor(Y_train_b).cuda()

                masks_pred = net(imgs) # output is a logit not a probability, -inf ~ inf
                masks_proba = F.sigmoid(masks_pred)

                masks_proba_flat = masks_proba.view(-1) # reshape, flatten

                true_masks_flat = true_masks.view(-1)

                loss = criterion(masks_proba_flat, true_masks_flat)
                epoch_loss += loss.item()

                print('{0:.4f} --- loss: {1:.6f}'.format(batch_no * batch_size / train_size, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch {} finished ! Loss: {}'.format(e, epoch_loss / (e + 1)))

            net  = net.eval() # evaluation mode
            val_dice = self.__eval_net(net, X_valid, Y_valid)
            print('Validation Dice Coeff: {}'.format(val_dice))

            torch.save(net.state_dict(), model_ckpt + '/unet{}.ckpt'.format(e))
            print('Checkpoint {} saved !'.format(e))

if __name__ == '__main__':
    ''''''
    # params
    parser = argparse.ArgumentParser()

    parser.add_argument('-phase', "--phase",
                        default= 'debug',
                        help= "project phase",
                        choices= ['train', 'debug', 'submit', 'resubmit'])
    parser.add_argument('-data_input', '--data_input',
                        default= '%s/raw' % (config.DataBaseDir)
                        )
    parser.add_argument('-model_output', '--model_output',
                        default= '%s/%s' % (config.ModelRootDir, config.strategy),
                        )
    args = parser.parse_args()

    #with utils.timer('Load raw data set'):
    train_data, image_files = data_utils.load_raw_train(args.data_input, return_image_files=True)

    print(len(train_data))

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data['images'].values, train_data['masks'].values, test_size= 0.2, random_state= 2018)
    print(len(X_train))
    print(len(X_valid))

    model = UNet_pt()
    model.fit(X_train, Y_train, X_valid, Y_valid, config.batch_size, 50, '.', 0.01)
