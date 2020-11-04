import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    """
    trainer without k-cross validation
    """

    def __init__(self, model, optimizer, trainset, testset, save_name, path, batch=32, num_workers=2,
                 criterion=nn.CrossEntropyLoss(), epoch=2):
        super(Trainer, self).__init__()

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
        self.batch = batch
        self.trainset = trainset
        self.testset = testset
        self.num_workers = num_workers
        self.save_name = save_name
        self.path = path
        self.epoch_loss = []
        self.epoch_acc = []

    def training(self, width=500):

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch,
                                                  shuffle=True, num_workers=self.num_workers)

        for e in range(self.epoch):

            mini_loss = 0.0
            mini_loss_list = []
            mini_acc_list = []

            for index, data in enumerate(trainloader, 0):
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

                # print statistics
                mini_loss += loss.item()

                if index % width == width - 1:
                    print('[epoch=%3d, batch_cnt=%5d] loss: %.5f' %
                          (e, index + 1, mini_loss / width))
                    mini_loss_list.append(mini_loss / width)
                    mini_loss = 0.0

            self.epoch_loss.append(mini_loss_list)
            self.saving(e)

    def testing(self):
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch,
                                                 shuffle=False, num_workers=self.num_workers)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        all = len(self.testset)
        print('Accuracy of the network on the {all} test images: %d %%' % (
                100 * correct / total))

    def saving(self, epoch=None):
        checkpoint = {
            'model_stat': self.model.state_dict(),
            'optimizer_stat': self.optimizer.state_dict(),
            'loss': self.criterion.state_dict(),
        }
        path = self.path + self.save_name

        if epoch is not None:
            path = path + '_epoch' + str(epoch)
        path += '.pth'

        torch.save(checkpoint, path)



