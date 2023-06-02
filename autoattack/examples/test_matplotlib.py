from PIL import Image
import matplotlib.pyplot as plt
'''
image1 = Image.open('C:\\Users\\ForYou\\Desktop\\svd_90_blenheim_spaniel_s_000006.PNG')
image2 = Image.open('C:\\Users\\ForYou\\Desktop\\svd_70_blenheim_spaniel_s_000006.PNG')
image3 = Image.open('C:\\Users\\ForYou\\Desktop\\svd_50_blenheim_spaniel_s_000006.PNG')
image4 = Image.open('C:\\Users\\ForYou\\Desktop\\svd_30_blenheim_spaniel_s_000006.PNG')
image5 = Image.open('C:\\Users\\ForYou\\Desktop\\svd_10_blenheim_spaniel_s_000006.PNG')
'''

image1 = Image.open('C:\\Users\\ForYou\\Desktop\\Figure_7.png')
image2 = Image.open('C:\\Users\\ForYou\\Desktop\\Figure_8.png')
image3 = Image.open('C:\\Users\\ForYou\\Desktop\\Figure_9.png')

image1_size = image1.size
image2_size = image2.size
image3_size = image3.size

cutout = Image.open('C:\\Users\\ForYou\\Desktop\\cutout_image.jpg')
cutmix = Image.open('C:\\Users\\ForYou\\Desktop\\cutmix_image.jpg')
mixup = Image.open('C:\\Users\\ForYou\\Desktop\\mixup_image.jpg')

'''
new_image = Image.new('RGB', (3 * image1_size[0], image1_size[1]), (32, 32, 32))
new_image.paste(image1, (0, 0))
new_image.paste(image2, (image1_size[0], 0))
new_image.paste(image3, (image1_size[0]*2, 0))
new_image.save("mixup_image.jpg", "JPEG")
new_image.show()
plt.imshow(new_image)
'''
fig = plt.figure()
rows = 3
cols = 1

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cutout)
ax1.set_title('Cutout')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(mixup)
ax2.set_title('Mixup')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(cutmix)
ax3.set_title('Cutmix')
ax3.axis("off")

plt.axis("off")

plt.show()
'''
fig = plt.figure()
rows = 1
cols = 5

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(image1)
ax1.set_title('SVD_90')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(image2)
ax2.set_title('SVD_70')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(image3)
ax3.set_title('SVD_50')
ax3.axis("off")

ax4 = fig.add_subplot(rows, cols, 4)
ax4.imshow(image4)
ax4.set_title('SVD_30')
ax4.axis("off")

ax5 = fig.add_subplot(rows, cols, 5)
ax5.imshow(image5)
ax5.set_title('SVD_10')
ax5.axis("off")

# plt.axis("off")
'''
plt.show()
'''
'''