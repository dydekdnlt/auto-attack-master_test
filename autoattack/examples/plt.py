import matplotlib.pyplot as plt

x = [i for i in range(0, 50)]

res_after = [45.85, 39.18, 67.33, 58.9, 74.78, 69.04, 74.24, 74.03, 70.83, 74.56, 89.4, 89.49, 89.48, 90.63, 89.14, 90.06, 89.54, 90.41, 90.64, 90.54, 89.53, 90.94, 90.17, 90.43, 90.33, 93.67, 93.33, 93.81, 93.76, 93.82, 93.77, 93.64, 93.88, 93.75, 93.81, 93.87, 93.92, 94.04, 93.88, 93.99, 94.22, 94.14, 94.19, 94.13, 94.2, 94.16, 94.15, 94.16, 94.22, 94.11]
# res_after = normalize x, adjust lr o
res_before = [43.93, 53.29, 61.15, 66.87, 70.3, 74.46, 77.23, 78.53, 80.85, 82.2, 83.48, 83.49, 84.62, 84.79, 84.77, 85.21, 86.43, 86.38, 86.53, 86.52, 86.91, 87.05, 87.19, 87.24, 87.52, 88.02, 87.68, 88.06, 88.11, 88.05, 88.23, 88.01, 88.59, 88.57, 88.67, 89.35, 89.07, 88.42, 88.7, 89.05, 88.46, 88.65, 89.24, 89.55, 89.51, 88.66, 89.11, 89.6, 89.58, 89.65]
# res_before = normalize o, adjust lr x
res_before2 = [61.44, 69.1, 72.28, 72.26, 74.4, 76.27, 74.91, 74.89, 71.59, 77.55, 89.13, 89.97, 90.43, 90.63, 90.35, 91.0, 90.66, 90.35, 90.19, 90.51, 88.08, 90.89, 90.35, 89.4, 89.99, 93.56, 93.72, 93.76, 93.89, 93.61, 93.84, 94.17, 94.04, 93.91, 94.16, 94.02, 93.98, 94.07, 94.11, 94.12, 94.08, 94.2, 94.23, 94.24, 94.1, 94.12, 94.26, 94.23, 94.25, 94.32]
# res_before2 = normalize o, adjust lr o
res_after2 = [44.59, 55.03, 62.44, 66.71, 71.1, 74.88, 76.52, 79.36, 80.06, 80.11, 82.71, 83.24, 84.23, 84.66, 84.7, 86.16, 85.97, 86.42, 86.88, 86.35, 86.93, 86.88, 87.05, 87.64, 87.17, 87.06, 88.1, 88.04, 87.18, 88.0, 88.02, 88.38, 87.93, 88.59, 88.03, 88.21, 88.21, 88.5, 88.89, 88.81, 88.86, 89.34, 88.97, 89.12, 89.75, 89.2, 89.43, 89.21, 89.1, 89.44]
# res_after2 = normalize x, adjust lr x
res_before2_svd_30 = [43.08, 52.89, 60.11, 66.63, 70.08, 72.68, 75.09, 76.57, 78.12, 78.43, 81.0, 81.43, 81.54, 81.51, 81.64, 81.76, 82.18, 82.31, 82.55, 82.19, 82.14, 82.48, 82.61, 82.7, 82.73, 82.88, 83.1, 82.82, 82.99, 83.08, 82.97, 83.03, 83.0, 82.99, 83.11, 82.95, 82.96, 83.21, 82.81, 83.03, 82.9, 83.05, 83.12, 82.93, 83.01, 82.91, 83.07, 83.09, 83.12, 82.98]

res_before2_svd_50 = [] # 얘 다시

res_before2_svd_30_test_only = [53.76, 54.26, 55.87, 59.79, 63.25, 62.08, 62.1, 63.68, 63.23, 62.59, 78.86, 78.61, 80.07, 80.5, 79.58, 79.61, 80.04, 80.18, 80.74, 79.9, 79.84, 80.43, 80.73, 81.14, 80.87, 84.64, 84.66, 84.9, 84.88, 85.28, 85.39, 85.05, 85.12, 85.38, 85.07, 84.98, 85.05, 84.72, 85.18, 84.75, 85.28, 84.99, 85.36, 85.4, 85.31, 85.27, 85.08, 85.23, 85.23, 85.39]
# cifar10
robust_res_after = []
robust_res_after2 = []
robust_res_before = []
robust_res_before2 = []
robust_res_before2_svd_30 = []
robust_res_before2_svd_50 = []
robust_res_before2_svd_30_test_only = []

# cifar100
'''
plt.plot(x, res, label='res')
# plt.plot(x, res_svd_10, label='res_svd_10')
# plt.plot(x, res_svd_30, label='res_svd_30')
# plt.plot(x, res_svd_50, label='res_svd_50')
# plt.plot(x, res_svd_70, label='res_svd_70')
plt.plot(x, res_svd_90, label='res_svd_90')
# plt.plot(x, res_svd_95, label='res_svd_95')
plt.plot(x, res_cut_mix, label='res_cut_mix')
plt.plot(x, res_mixup, label='res_mixup')
plt.plot(x, res_svd_90_mixup, label='res_svd_90_mixup')
plt.plot(x, res_svd_90_cut_mix, label='res_svd_90_cutmix')
plt.xlabel('number of epoch')
plt.ylabel('accuracy')
plt.title('Comparison of SVD, Mixup, CutMix | backbone=Resnet')
plt.legend(loc='lower right')
plt.show()
'''
'''
plt.plot(x, nf, label='nf')
# plt.plot(x, nf_svd_10, label='nf_svd_10')
# plt.plot(x, nf_svd_30, label='nf_svd_30')
plt.plot(x, nf_svd_50, label='nf_svd_50')
plt.plot(x, nf_svd_70, label='nf_svd_70')
plt.plot(x, nf_svd_90, label='nf_svd_90')
plt.plot(x, nf_svd_95, label='nf_svd_95')
plt.plot(x, nf_cut_mix, label='nf_cut_mix')
plt.plot(x, nf_mixup, label='nf_mixup')
plt.xlabel('number of epoch')
plt.ylabel('accuracy')
plt.title('Comparison of SVD, Mixup, CutMix | backbone=Nfnet')
plt.legend(loc='lower right')
plt.show()
'''