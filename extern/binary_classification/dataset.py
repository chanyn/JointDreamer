import os
import pickle
import random

import numpy as np
import zipfile
import torch
from torch.utils.data import Dataset
import cv2
from functools import reduce
import itertools
from PIL import Image

from threestudio.utils.misc import cleanup
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
try:
    import pyspng
except ImportError:
    pyspng = None



class IDPoseDataset(Dataset):
    def __init__(
            self,
            path,  # Path to directory or zip.
            path_pkl,
            negative_pkl,
            neg_max=True,
            test=False,
            disc_cls='all',
            resolution=224,
            transform=None,
            camera_normalize=False,
            num_positive_pairs=1000,
    ):
        self.img_size = resolution
        self.root = path
        self.disc_cls = disc_cls
        self.transform = transform
        self.camera_normalize = camera_normalize

        self.obj_list = pickle.load(open(path_pkl, 'rb'))
        self.ids_list = list(itertools.combinations([str(i).zfill(3) for i in range(12)], 2))
        positive_list = list(itertools.product(range(len(self.obj_list)), self.ids_list))
        positive_list = random.sample(positive_list, num_positive_pairs) # reduce data
        negative_list = pickle.load(open(negative_pkl, 'rb'))
        if neg_max:
            negative_list = random.sample(negative_list, len(positive_list))

        pair_list = []
        for pair in positive_list:
            pair_list.append(list(pair) + [1.])
        if disc_cls in ['id', 'all']:
            for pair in negative_list:
                pair_list.append(list(pair) + [0.])
        if test:
            self.pair_list = random.sample(pair_list, min(5000, len(pair_list)))
        else:
            self.pair_list = pair_list

        # name = os.path.splitext(os.path.basename(self.root))[0]
        print(
            '==> use image path: %s, num images: %d' % (self.root, len(self.pair_list)))

    def load_image(self, obj, id):
        image_path = os.path.join(self.root, self.obj_list[obj], id + '.png')
        image = Image.open(image_path).convert("RGB")
        # image = np.array(image).astype(np.float32) / 255.0  # 0~1
        # resize_img = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # img = resize_img.transpose(2, 0, 1)
        return image

    def load_delta_transform(self, obj, id):
        assert isinstance(obj, tuple)
        npy_path_1 = os.path.join(self.root, self.obj_list[obj[0]], id[0] + '.npy')
        npy_path_2 = os.path.join(self.root, self.obj_list[obj[1]], id[1] + '.npy')

        matrix_1 = np.concatenate(
            [np.load(npy_path_1), np.array([[0, 0, 0, 1]])], axis=0)  # 3*4
        matrix_2 = np.concatenate(
            [np.load(npy_path_2), np.array([[0, 0, 0, 1]])], axis=0)  # 3*4
        delta_transform = reduce(np.dot, [matrix_1, matrix_2.T])
        return delta_transform

    def find_mismatch_delta(self, obj, id, true_delta):
        while True:
            id_ = random.choice([str(i).zfill(3) for i in range(12)])
            delta_matrix = self.load_delta_transform(obj, (id[0], id_))
            if not (delta_matrix == true_delta).all():
                break
        return delta_matrix

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        fname = self.pair_list[idx]
        path_1, path_2, label = fname

        if label == 1:
            img1 = self.load_image(path_1, path_2[0])
            img2 = self.load_image(path_1, path_2[1])
            delta_pair = path_2
            obj_pair = (path_1, path_1)
        elif label == 0:
            delta_pair = random.choice(self.ids_list)
            img1 = self.load_image(path_1, delta_pair[0])
            img2 = self.load_image(path_2, delta_pair[1])
            obj_pair = (path_1, path_2)
        else:
            raise ValueError('label error')


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        delta_transform = self.load_delta_transform(obj_pair, delta_pair)
        if self.camera_normalize:
            scale = np.linalg.norm(delta_transform)
            scale = np.sqrt(scale)
            delta_transform = delta_transform / scale

        if random.random() > 0.5 and self.disc_cls in ['all', 'dir']: # wrong pose
            delta_transform = self.find_mismatch_delta(obj_pair, delta_pair, delta_transform)
            label = 0.

        sample = {'img1': img1,
                  'img2': img2,
                  'transform': delta_transform.astype(np.float32),
                  'label': np.array([label]).astype(np.float32)}
        return sample


class IDPoseReconDataset(IDPoseDataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        path_pkl,
        negative_pkl,
        neg_max=True,
        test=False,
        disc_cls='all',
        resolution=224,
        transform=None,
        camera_normalize=False,
        num_positive_pairs=1000,
        diffusion='stabilityai/stable-diffusion-2-1-base'
    ):
        super(IDPoseReconDataset, self).__init__(
            path, path_pkl, negative_pkl, neg_max, test, disc_cls, resolution, transform, camera_normalize, num_positive_pairs)

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(diffusion, **pipe_kwargs).to('cuda')
        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)



if __name__ == "__main__":
    dataset = IDPoseDataset(path='/home/dycpu3_8tssd/jch/data/views_release',
                            path_pkl='/home/dycpu3_8tssd/jch/data/obj_path.pkl',
                            negative_pkl='/home/dycpu3_8tssd/jch/data/negative_pairs.pkl',
                            )
    for i in range(len(dataset)):
        data = dataset[i]
        print(data[0].shape, data[1].shape, data[2]) # 6x512x512, 4x4, 0/1
