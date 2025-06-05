import os
import numpy as np
from PIL import Image
import random

height = 256
width = 256
channels = 3

random_seed = 42

images_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'your_dataset', 'images')
masks_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'your_dataset', 'masks')

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
random.Random(random_seed).shuffle(image_files)

all_imgs = []
all_masks = []
for fname in image_files:
    img = Image.open(os.path.join(images_dir, fname)).convert('RGB').resize((width, height))
    mask_name = os.path.splitext(fname)[0] + '_mask.png'
    mask = Image.open(os.path.join(masks_dir, mask_name)).convert('L').resize((width, height))
    mask = np.array(mask, dtype=np.float32)
    mask = np.where(mask > 0, 255.0, 0.0)
    all_imgs.append(np.asarray(img, dtype=np.float32))
    all_masks.append(mask)

all_imgs = np.stack(all_imgs)
all_masks = np.stack(all_masks)

all_imgs = (all_imgs - all_imgs.min()) / (all_imgs.max() - all_imgs.min()) * 255

n_total = len(all_imgs)
train_n = int(0.6 * n_total)
val_n = int(0.2 * n_total)

train_img = all_imgs[:train_n]
val_img = all_imgs[train_n:train_n+val_n]
test_img = all_imgs[train_n+val_n:]

train_mask = all_masks[:train_n]
val_mask = all_masks[train_n:train_n+val_n]
test_mask = all_masks[train_n+val_n:]

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'pennfudan_npy')
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, 'data_train.npy'), train_img)
np.save(os.path.join(out_dir, 'data_val.npy'), val_img)
np.save(os.path.join(out_dir, 'data_test.npy'), test_img)
np.save(os.path.join(out_dir, 'mask_train.npy'), train_mask)
np.save(os.path.join(out_dir, 'mask_val.npy'), val_mask)
np.save(os.path.join(out_dir, 'mask_test.npy'), test_mask)

print('Prepared dataset saved to', out_dir)
