import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import timm
import sys
import torchvision
from torchvision.datasets import ImageFolder

sys.path.insert(0, '..')

from nf_resnet import *
from resnet import *
# from resnet_gray import *
# from Resnet_elu import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8. / 255.) # 8 or 16 변경
    parser.add_argument('--model', type=str, default='./pt/new_resnet_test_swa.pt')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def filter_state_dict(state_dict):
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']

        return state_dict
    # load model
    model = ResNet50()
    # print('model : ', model, sep='\n')

    # model = NFNet(num_classes=10, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5, activation='gelu')
    model.to(device)
    # ckpt = torch.load(args.model)
    ckpt = filter_state_dict(torch.load(args.model))

    model.load_state_dict(ckpt)
    # model.cuda()

    model.eval()

    # load data
    # transform_list = [transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)

    # svd, mixup 전용 normalize 
    transform_chain = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10 전용
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # mnist 전용
        ])
    '''
    # cutmix, cutout 전용 normalize
    transform_chain = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
    '''
    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=False)
    item = ImageFolder('./../../data/test', transform=transform_chain)
    test_loader = data.DataLoader(item, batch_size=16, shuffle=False, num_workers=2)
    # print(item.classes, item.class_to_idx)
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack    
    from autoattack import AutoAttack

    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
                           version=args.version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    # print(y_test, y_test.size())
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                             bs=args.batch_size, label=item.classes)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                                                                        y_test[:args.n_ex], bs=args.batch_size,
                                                                        label=item.classes)

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
