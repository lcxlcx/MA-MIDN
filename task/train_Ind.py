import torch
from model.MA_MIDN import MA_MIDN
import torch.optim as optim
from utils import AverageMeter,getWorkBook
import time
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from scipy.misc import imsave
import os

class Trainer_Ind(object):

    def __init__(self, config, data_loader):

        self.config = config

        if config.is_train:
            self.train_loader = data_loader[0]
            self.test_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_test = len(self.test_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.epochs = config.epochs
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.step_size = config.step_size

        self.mywb = getWorkBook()
        self.ckpt_dir = '/home'
        self.best_valid_accs = 0.

        self.use_gpu = config.use_gpu
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('Init Model')
        self.model = MA_MIDN()

        if torch.cuda.is_available():
            self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

        print('[*] Number of parameters of model: {:,}'.format(

            sum([p.data.nelement() for p in self.model.parameters()])))



    def train(self):


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_test)
        )

        for epoch in range(self.epochs):

            self.scheduler.step(epoch)

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)
            test_losses, test_accs = self.test(epoch)

            is_best = test_accs.avg > self.best_valid_accs
            self.best_valid_accs = max(test_accs.avg, self.best_valid_accs)

            msg1 = "model_: train loss: {:.4f} - train acc: {:.4f} "
            msg2 = "- test loss: {:.4f} - test acc: {:.4f}"
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, test_losses.avg, test_accs.avg))

            self.save_checkpoint(epoch,
                                 {'epoch': epoch + 1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_valid_acc': self.best_valid_accs,
                                  }, is_best
                                 )
            self.record_loss_acc(train_losses.avg, train_accs.avg, test_losses.avg, test_accs.avg)
            dir = "/home/train.xlsx"
            self.mywb.save(dir)

    def train_one_epoch(self, epoch):

        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        self.model.train()

        tic = time.time()

        with tqdm(total=self.num_train) as pbar:
            for i, (bag_data, label) in enumerate(self.train_loader):
                bag_label = label[0]
                if torch.cuda.is_available():
                    bag_data, bag_label = bag_data.to(self.device), bag_label.to(self.device)
                bag_data, bag_label = Variable(bag_data), Variable(bag_label)
                # bag_data = bag_data.squeeze(0)

                loss, _ = self.model.calculate_objective(bag_data, bag_label)
                losses.update(loss.data[0].item(), 1)
                error, predicted_label_train = self.model.calculate_classification_error(bag_data, bag_label)
                correct_label_pred = (int(bag_label) == int(predicted_label_train))
                if correct_label_pred:
                    correct_count = 1
                else:
                    correct_count = 0

                accs.update(correct_count, 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model_train_loss: {} - model_train_acc: {:.6f}".format(
                            (toc - tic), losses.avg, accs.avg
                        )
                    )
                )
                self.batch_size = 1
                pbar.update(self.batch_size)

        return losses, accs

    def test(self, epoch):

        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        self.model.eval()

        tic = time.time()

        with tqdm(total=self.num_test) as pbar:
            for i, (bag_data, label) in enumerate(self.test_loader):
                bag_label = label[0]
                if torch.cuda.is_available():
                    bag_data, bag_label = bag_data.to(self.device), bag_label.to(self.device)
                bag_data, bag_label = Variable(bag_data), Variable(bag_label)

                loss, score = self.model.calculate_objective(bag_data, bag_label)
                losses.update(loss.data[0].item(), 1)
                error, predicted_label_train = self.model.calculate_classification_error(bag_data, bag_label)
                correct_label_pred = (int(bag_label) == int(predicted_label_train))
                if correct_label_pred:
                    correct_count = 1
                else:
                    correct_count = 0

                accs.update(correct_count, 1)


                visualization_attentionA(bag_data[0], score[0], i, epoch)
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model_test_loss: {} - model_test_acc: {:.6f}".format(
                            (toc - tic), losses.avg, accs.avg
                        )
                    )
                )
                self.batch_size = 1
                pbar.update(self.batch_size)
        return losses, accs

    def save_checkpoint(self, i, state, is_best):
        if is_best:
            filename = 'AMIL_100x_best.pth.tar'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
            torch.save(state, ckpt_path)


    def record_loss_acc(self, epoch_trainloss, epoch_trainacc, epoch_testloss, epoch_testacc):
        self.mywb["epoch_trainloss"].append([epoch_trainloss])
        self.mywb["epoch_trainacc"].append([epoch_trainacc])
        self.mywb["epoch_testloss"].append([epoch_testloss])
        self.mywb["epoch_testacc"].append([epoch_testacc])


def visualization_attention(data, attention_weights, batch_idx, epoch):
    img_save_dir = './AMIL_visualization/BreaKHis/epoch_{}'.format(epoch)
    img_save_name = img_save_dir + '/test_epoch_{}_no_{}.png'.format(epoch, batch_idx)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    data = data.cpu().data.numpy()
    attention_weights = attention_weights.cpu().data.numpy()
    # print("data.shape",data.shape)
    # print("attention_weights",attention_weights.shape)
    attention_weights = attention_weights / np.max(attention_weights)
    complete_image = np.zeros((3, 448, 700))
    for height_no in range(16):
        for width_no in range(25):
            complete_image[:,height_no*28:height_no*28+28, width_no*28:width_no*28+28] = data[height_no*25+width_no,:,:,:] * attention_weights[height_no*25+width_no]
    complete_image = complete_image.transpose((1, 2, 0))
    imsave(img_save_name, complete_image)

