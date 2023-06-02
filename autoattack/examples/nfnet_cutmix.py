import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from nf_resnet import NFNet
from optimizer import SGD_AGC
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
best_acc = 0
accuracy_list = []


def run():
    parser = argparse.ArgumentParser(description='NFNet CutMix Training')
    parser.add_argument('--variant', default='F0', type=str, choices=['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'],
                        help='NFNet variants')
    parser.add_argument('--lr', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='the number of the epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch sizes')
    parser.add_argument('--beta', default=1, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')
    parser.add_argument('--alpha', default=300, type=float, help='number of new channel increases per depth')
    parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_dataset = ImageFolder("C:\\Users\\ForYou\\Desktop\\auto-attack-master\\data\\train",
                                transform=transform_train)

    test_dataset = ImageFolder('C:\\Users\\ForYou\\Desktop\\auto-attack-master\\data\\test',
                               transform=transform_test)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=2)

    model = NFNet(num_classes=10, variant=args.variant, stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5,
                  activation='gelu').to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD_AGC(named_params=model.named_parameters(), lr=args.lr, momentum=0.9, clipping=0.1,
                        weight_decay=5e-4,
                        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def train(epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # new_images = images.cpu().numpy()
                # plt.imshow((new_images[0].T * 255).astype(np.uint8))
                # plt.show()
                outputs = model(images)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, args.num_epochs, i + 1,
                                                                      len(train_loader),
                                                                      loss.item()))

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
            torch.save({'state_dict': model.state_dict()}, 'nfnet_test_cut_mix.pt')
            best_acc = acc
            print('Best Accuracy : {} %'.format(best_acc))

    for epoch in range(args.num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        print(accuracy_list)


if __name__ == '__main__':
    run()
    print("accuracy_list : ", accuracy_list)
