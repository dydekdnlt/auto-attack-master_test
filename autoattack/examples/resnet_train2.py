import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from optimizer import SGD_AGC
from torchvision.datasets import ImageFolder
from resnet import ResNet50
# from Resnet_elu import ResNet50
# from resnet_gray import ResNet50
import trades
from mart import mart_loss
from hat import hat_loss
import random
import torch.nn.functional as F

best_acc = 0
accuracy_list = []


def run():
    # Hyper-parameters
    parser = argparse.ArgumentParser(description='ResNet Training')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--num-steps', default=10, help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007, help='perturb step size')
    parser.add_argument('--beta', default=5.0, help='regularization, i.e., 1/lambda in TRADES')  # range=(1, 5)
    parser.add_argument('--gamma', default=1.0, type=float, help='Weight of helper loss in HAT.')
    parser.add_argument('--h', default=2.0, type=float,
                        help='Parameter h to compute helper examples (x + h*r) for HAT.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image preprocessing modules
    transform_train = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # CIFAR-10 dataset
    # train_dataset = torchvision.datasets.CIFAR10(root='autoattack\\examples', train=True, transform=transform_train, download=False)

    # test_dataset = torchvision.datasets.CIFAR10(root='autoattack\\examples', train=False, transform=transform_test)

    # CIFAR-10-svd dataset

    train_dataset = ImageFolder("./../../data/new_train_pca_10",
                                transform=transform_train)

    test_dataset = ImageFolder('./../../data/test',
                               transform=transform_test)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,
                                              num_workers=2)
    # Model
    model = ResNet50().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD_AGC(named_params=model.named_parameters(), lr=args.lr, momentum=0.9, clipping=0.1,
                        weight_decay=5e-4,
                        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def adjust_learning_rate(optimizer, epoch):

        lr = args.lr
        if epoch >= 10:
            lr = args.lr * 0.1

        if epoch >= 25:
            lr = args.lr * 0.01

        if epoch >= 40:
            lr = args.lr * 0.001

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Train the model
    def train(epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            # print(images.shape)

            optimizer.zero_grad()
            # Normal
            loss = criterion(model(images), labels)

            '''
            # TRADES
            loss = trades.trades_loss(model=model,
                                      x_natural=images,
                                      y=labels,
                                      optimizer=optimizer,
                                      step_size=args.step_size,
                                      epsilon=args.epsilon,
                                      perturb_steps=args.num_steps,
                                      beta=args.beta,
                                      distance='l_inf')

            # MART
            '''
            '''
            loss = mart_loss(model=model,
                             x_natural=images,
                             y=labels,
                             optimizer=optimizer,
                             step_size=args.step_size,
                             epsilon=args.epsilon,
                             perturb_steps=args.num_steps,
                             beta=args.beta)
            '''
            # HAT
            '''


            loss = hat_loss(model=model,
                            x=images,
                            y=labels,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            h=args.h,
                            beta=args.beta,
                            gamma=args.gamma)
            '''
            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, 50, i + 1,
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
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Epoch [{}/{}], Accuracy of the model on the test images: {} %'.format(epoch + 1, 50,
                                                                                         100 * correct / total))

        acc = 100 * correct / total
        accuracy_list.append(acc)
        if acc > best_acc:
            # Save the model checkpoint
            torch.save({'state_dict': model.state_dict()}, './pt/new_resnet_test_pca_10.pt')
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
# 완료 후 mnist 50, 90 훈련