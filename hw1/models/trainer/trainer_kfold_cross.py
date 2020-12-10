import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .trainer import Trainer

import numpy as np
import json


class TrainerWithKFoldCrossValidation(Trainer):
    def __init__(self, model, optimizer, trainset, testset, save_name, path, kfold=5, batch=16, num_workers=2,
                 criterion=nn.CrossEntropyLoss(), epoch=2):
        super(TrainerWithKFoldCrossValidation, self).__init__(model, optimizer, trainset, testset, save_name, path,
                                                              batch, num_workers, criterion, epoch)
        self.kfold = kfold
        self.epoch_loss = []
        self.epoch_acc = []
        self.epoch_tacc = []

    def training(self, width=500):
        cnt = len(self.trainset)

        split = int(cnt/self.kfold)   # ceiling
        while (cnt-split) % (self.kfold-1) != 0:
            split -= 1

        remain = cnt - split
        train_cnt = []
        for i in range(1, self.kfold):
            train_cnt = train_cnt + [int(remain / self.kfold)]

        cut = [split] + train_cnt
        split_trainset = torch.utils.data.random_split(self.trainset, cut)

        for e in range(1, self.epoch + 1):

            epoch_acc_list = []
            epoch_loss_list = []
            epoch_tacc_list = []


            # For each k validation
            for valid_index, validation_set in enumerate(split_trainset, 0):

                kth_loss_list = []
                kth_acc_list = []
                kth_tacc_list = []


                valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch, shuffle=True,
                                                           num_workers=self.num_workers)

                for train_index, train_set in enumerate(split_trainset, 0):

                    if train_index == valid_index:
                        continue

                    train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch, shuffle=True,
                                                               num_workers=self.num_workers)

                    mini_train_loss = 0.0  # kth mini batch loss
                    mini_batch = 0

                    mini_loss_list = []
                    mini_acc_list = []
                    mini_tacc_list = []

                    for index, data in enumerate(train_loader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        loss.backward()
                        self.optimizer.step()

                        mini_batch += 1
                        mini_train_loss += loss.item()

                        if mini_batch % width == width - 1:  # print every 500 mini-batches
                            print('[epoch=%3d, fold=%3d, batch_cnt=%5d] loss: %.5f' %
                                  (e, train_index, mini_batch + 1, mini_train_loss / width))
                            mini_loss_list.append(mini_train_loss / width)

                            mini_train_loss = 0.0

                            with torch.no_grad():
                                correct = 0
                                total = 0
                                for data in valid_loader:
                                    images, labels = data
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    outputs = self.model(images)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()

                                mini_tacc_list.append(100 * correct / total)
                                print('Train Acc: %d %%' % (100 * correct / total))

                                correct = 0
                                total = 0
                                for data in valid_loader:
                                    images, labels = data
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    outputs = self.model(images)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()

                                mini_acc_list.append(100 * correct / total)
                                print('Accuracy: %d %%' % (100 * correct / total))

                    kth_loss_list.append(mini_loss_list)
                    kth_acc_list.append(mini_acc_list)

                    kth_tacc_list.append((mini_tacc_list))

                epoch_loss_list.append(kth_loss_list)
                epoch_acc_list.append(kth_acc_list)
                epoch_tacc_list.append(kth_tacc_list)

            self.epoch_acc.append(epoch_acc_list)
            self.epoch_loss.append(epoch_loss_list)
            self.epoch_tacc.append(epoch_tacc_list)

        self.saving()

        epoch_acc = np.array(self.epoch_acc).flatten()
        epoch_loss = np.array(self.epoch_loss).flatten()
        epoch_tacc = np.array(self.epoch_tacc).flatten()

        with open('epoch.json', 'w') as fp:
            epoch_info = {
                'acc': list(epoch_acc),
                'loss': list(epoch_loss),
                'tacc': list(epoch_tacc)
            }
            json.dump(epoch_info, fp)


        print('Finished Training')