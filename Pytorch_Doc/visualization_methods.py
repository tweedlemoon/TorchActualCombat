# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail.png"

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    '''
    展示图片的函数，下面有示例
    :param imgs: grid of the image sets
    :return: None
    '''
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


datasets_path = '../Datasets/Visualization_example'
dog1_int = read_image(str(datasets_path + '/dog1.jfif'))
# dog2_int = read_image(str(datasets_path + '/dog2.jfif'))
dog1_size = dog1_int.shape
grid = make_grid([dog1_int])
# grid = make_grid([dog1_int, dog2_int, dog1_int, dog2_int])
show(grid)
plt.show()

from torchvision.transforms.functional import convert_image_dtype

batch_int = torch.stack([dog1_int, ])
batch = convert_image_dtype(batch_int, dtype=torch.float)

from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(pretrained=True, progress=False)
print(model)
model = model.eval()

normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
output = model(normalized_batch)['out']
print(output.shape, output.min().item(), output.max().item())

sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

normalized_masks = torch.nn.functional.softmax(output, dim=1)

dog_and_boat_masks = [
    normalized_masks[img_idx, sem_class_to_idx[cls]]
    for img_idx in range(batch.shape[0])
    for cls in ('dog', 'boat')
]

show(dog_and_boat_masks)
plt.show()