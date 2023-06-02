import math
import time

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as F


from other_utils import Logger
import checks

import cv2
from matplotlib import pyplot as plt
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SVD:
    def get(self, A, compRate = 0.9):
        n = A.shape[0]
        p = A.shape[1]
        # A = U(n,n) * diag(s(n)) * V(n,p)
        # full_matrices=False 옵션에 의해 "0"인 부분을 삭제하고 리턴
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        k = int(compRate * (n*p) / (n+1+p)) # k: 고유값 사용갯수
        S = np.diag(s[:k])
        B = np.dot(U[:, :k], np.dot(S, VT[:k, :]))
        B = (255*(B - np.min(B))/np.ptp(B)).astype(np.uint8)
        return B, U[:, :k], s[:k], VT[:k, :], k


class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            from autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)
            
            from fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                logger=self.logger)
    
        else:
            from autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
            
            from fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
    
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def get_svd(self, img, rate):
        n = img.shape[0]
        p = img.shape[1]

        U, s, VT = np.linalg.svd(img, full_matrices=False)
        k = int(rate * (n * p) / (n + 1 + p))  # k: 고유값 사용갯수
        S = np.diag(s[:k])
        B = np.dot(U[:, :k], np.dot(S, VT[:k, :]))
        B = (255 * (B - np.min(B)) / np.ptp(B)).astype(np.uint8)
        return B, U[:, :k], s[:k], VT[:k, :], k

    def get_svd_tensor(self, x):

        tf_toPILImage = ToPILImage()
        tf_toTensor = ToTensor()
        new_x = x.clone()
        for i in range(x.size()[0]):
            image = x[i, :, :, :].cpu().numpy()
            # print(image)
            image = np.transpose(image, (1, 2, 0))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            image = np.clip(255.0 * image, 0, 255)
            image = image.astype(np.uint8)
            img = tf_toPILImage(image)
            # plt.imshow(img)
            # plt.show()
            # image = PILimage(x[i, :, :, :])
            img = np.float32(img)
            img = np.array(img)
            # print(img)
            R, _, _, _, _ = self.get_svd(img[:, :, 0], 0.1)
            G, _, _, _, _ = self.get_svd(img[:, :, 1], 0.1)
            B, _, _, _, _ = self.get_svd(img[:, :, 2], 0.1)

            newImg = np.zeros_like(img)
            newImg[:, :, 0] = R
            newImg[:, :, 1] = G
            newImg[:, :, 2] = B
            # print(newImg, newImg.shape)
            newImg = np.clip(newImg / 255, 0, 1)
            new_x[i, :, :, :] = tf_toTensor(newImg)
            # new_x[i, :, :, :] = F.to_tensor(newImg)
            '''
            newImg2 = newImg.astype(np.uint8)
            newImg2 = Image.fromarray(newImg2)
            plt.imshow(newImg2)
            plt.show()
            '''

        # print(x.size(), new_x.size())
        # new_x = new_x.type(torch.FloatTensor)
        '''
        new_x_numpy = new_x.cpu().numpy()
        new_x_2 = np.clip(new_x_numpy / 255, 0, 1)
        new_x_3 = F.to_tensor(new_x_2).to(self.device)
        '''
        # print(new_x, new_x.size())
        return new_x

    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False, label=[]):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        # print("test : ", label)
        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)

        with torch.no_grad():
            # calculate accuracy
            # print(x_orig, y_orig)
            # print('=============')
            # print(x_orig.size(), y_orig.size())

            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            # robust_flags2 = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            y_adv = torch.empty_like(y_orig)
            # y_adv2 = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                # robust_flags2[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            # robust_accuracy2 = torch.sum(robust_flags2).item() / x_orig.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}
            # robust_accuracy_dict2 = {'clean': robust_accuracy2}

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            # x_adv2 = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    # print(x.size())
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    # x_svd90 = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    # y_svd90 = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    # x_svd90 = SVD.get()
                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    '''
                    adv_curr = self.apgd.perturb(x, y) # 기존 perturb code
                    
                    x_svd = SVD(x) # 각 batch의 image svd 적용 definition function 생성
                    
                    adv_curr_svd = self.apgd.perturb(x_svd, y) # (x_svd + noise) image의 batch
                    
                    x_svd - adv_curr_svd = noise_2 # noise만 추출
                    
                    adv_curr_2 = x + noise_2 # 원본 이미지 x + noise
                    '''
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)  # adv_curr은 원본 이미지 텐서 x를 공격한 이미지 텐서
                        # x_svd90 = self.get_svd_tensor(x).to(self.device)  # x_svd90은 텐서 x를 svd 변환한 텐서
                        # adv_curr_svd90 = self.apgd.perturb(x_svd90, y)  # adv_curr_svd90은 x_svd90을 공격한 텐서
                        # noise = adv_curr_svd90 - x_svd90  # noise는 svd 텐서의 노이즈를 갖는 변수
                        # new_x = (x + noise).to(self.device)  # new_x는 원본 x에 svd텐서의 노이즈를 더한 것

                        # noise 다시 수정
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                        #print(adv_curr.size())
                        #print(x.size())
                        #print(adv_curr.size()[0])
                        '''
                        CIFAR100_MEAN, CIFAR100_STD = np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])
                        for i in range(adv_curr.size()[0]):
                            image = adv_curr[i, :, :, :].cpu().numpy()
                            image = np.transpose(image, (1, 2, 0))
                            image = np.clip(255.0 * (image * CIFAR100_STD + CIFAR100_MEAN), 0, 255)
                            # print(x[i, :, :, :].size(), label[y[i]])
                            # image = PILimage(x[i, :, :, :])
                            image = np.float32(image)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = image.astype(np.uint8)
                            # print(type(image), label[y[i]])
                            # image = torchvision.transforms.ToPILImage(image)
                            # print(image.size())
                            # plt.imshow(image)
                            # plt.show()
                            # cv2.imwrite('C:\\Users\\ForYou\\Desktop\\auto-attack-master\\data\\train_image_cifar100_adv\\{}\\{}.png'.format(label[y[i]], time.time_ns()), image)
                        '''
                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    # output2 = self.get_logits(new_x).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    # false_batch2 = ~y.eq(output2).to(robust_flags2.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    # non_robust_lin_idcs2 = batch_datapoint_idcs[false_batch2]
                    robust_flags[non_robust_lin_idcs] = False
                    # robust_flags2[non_robust_lin_idcs2] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # x_adv2[non_robust_lin_idcs2] = new_x[false_batch2].detach().to(x_adv2.device)

                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)
                    # y_adv2[non_robust_lin_idcs2] = output2[false_batch2].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        # num_non_robust_batch2 = torch.sum(false_batch2)
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed(original noise)'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                        # self.logger.log('{} - {}/{} - {} out of {} successfully perturbed(svd noise)'.format(
                            # attack, batch_idx + 1, n_batches, num_non_robust_batch2, x.shape[0]))
                    # 해당 부분에서 공격 성공시 비교 코드 작성
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                # robust_accuracy2 = torch.sum(robust_flags2).item() / x_orig.shape[0]
                # robust_accuracy_dict2[attack] = robust_accuracy2
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s(original))'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    # self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s(svd))'.format(
                        # attack.upper(), robust_accuracy2, time.time() - startt))
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

