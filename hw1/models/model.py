import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.optim as optim
from .vgg16 import _Vgg16, Block
from .trainer.trainer_kfold import TrainerWithKFoldValidation

BATCH_SIZE = 32
EPOCH = 50

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



vgg_16_adam = _Vgg16(Block, [2, 2, 3, 3, 3])
summary(vgg_16_adam, (3, 64, 64), device='cuda')
optimizer_adam = optim.Adam(vgg_16_adam.parameters(), lr=0.001)

trainer = TrainerWithKFoldValidation(vgg_16_adam, optimizer_adam, trainset, testset, save_name="vgg16_adam_batch32_fold5", path="/weight", epoch=EPOCH, batch=BATCH_SIZE)