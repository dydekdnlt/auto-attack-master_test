import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse

from torchvision.transforms import ToPILImage

from resnet import ResNet50
from optimizer import SGD_AGC
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

best_acc = 0
accuracy_list = []


def run():
    parser = argparse.ArgumentParser(description='ResNet Mixup Training')
    parser.add_argument('--lr', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='the number of the epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch sizes')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default = 1)')
    parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency')
    parser.add_argument('--resume', '-r', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_dataset = ImageFolder("./../../data/mnist_training",
                                transform=transform_train)

    test_dataset = ImageFolder('./../../data/mnist_testing',
                               transform=transform_test)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=2)

    model = ResNet50().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD_AGC(named_params=model.named_parameters(), lr=args.lr, momentum=0.9, clipping=0.1,
                        weight_decay=5e-4,
                        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def mixup_data(x, y, alpha=1.0):

        # tf_toPILImage = ToPILImage()

        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        '''
        for i in range(0, batch_size):
            print(y_a[i], y_b[i])
            image = mixed_x[i, :, :, :].cpu().numpy()
            # print(image)
            image = np.transpose(image, (1, 2, 0))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            image = np.clip(255.0 * image, 0, 255)
            image = image.astype(np.uint8)
            img = tf_toPILImage(image)
            plt.imshow(img)
            plt.show()
        '''
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):

        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train(epoch):
        model.train()
        tf_toPILImage = ToPILImage()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            a = random.randint(0, images.size()[2] - 1)
            b = round(random.uniform(-8. / 255., 8. / 255.), 4)
            # print(a, images.size())
            images[:, :, a, :] += b

            images, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha)
            images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))
            '''
            image = images[a, :, :, :].cpu().numpy()
            # print(image)
            image = np.transpose(image, (1, 2, 0))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            image = np.clip(255.0 * image, 0, 255)
            image = image.astype(np.uint8)
            img = tf_toPILImage(image)
            plt.imshow(img)
            plt.show()
            '''
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Backward and optimize

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, args.num_epochs, i + 1,
                                                                      len(train_loader),
                                                                      loss.item()))

    def val(epoch):
        # global best_acc, accuracy_list
        model.eval()
        with torch.no_grad():
            val_correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            print('Epoch [{}/{}], Accuracy of the model on the train images: {} %'.format(epoch + 1, 50,
                                                                                         100 * val_correct / total))

    # Test the model
    def test(epoch):
        global best_acc, accuracy_list
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Epoch [{}/{}], Accuracy of the model on the test images: {} %'.format(epoch + 1, args.num_epochs,
                                                                                         100 * correct / total))

        acc = 100 * correct / total
        accuracy_list.append(acc)
        if acc > best_acc:
            # Save the model checkpoint
            # torch.save(model.state_dict(), 'nf_model.pt')
            torch.save({'state_dict': model.state_dict()}, './pt/resnet_test_mnist_mixup.pt')
            best_acc = acc
            print('Best Accuracy : {} %'.format(best_acc))

    for epoch in range(50):
        # adjust_learning_rate(optimizer, epoch)
        train(epoch)
        val(epoch)
        test(epoch)
        scheduler.step()
        print(accuracy_list)


if __name__ == '__main__':
    run()
    print("accuracy_list : ", accuracy_list)
