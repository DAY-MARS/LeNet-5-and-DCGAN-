import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(10,10))
imgs = ['epoch_5.png', 'epoch_10.png', 'epoch_20.png', 'epoch_30.png']
titles = ['Epoch 5', 'Epoch 10', 'Epoch 20', 'Epoch 30']
for ax, img, title in zip(axes.flat, imgs, titles):
    ax.imshow(mpimg.imread(img))
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig('combined.png')
plt.show()