from functools import partial
from glob import glob
from torch.utils import data
from monai import transforms as T
import nibabel as nib
import numpy as np
import pandas as pd
import os
import random, csv
import torch

# from torchsampler import ImbalancedDatasetSampler


class DataFolder(data.Dataset):
	def __init__(self, image_dir, image_type, transform, mode='train'):
		self.__image_reader = {
			'np': lambda url: np.load(url),
			'nii': lambda url: nib.load(url).get_fdata()
		}
		self.image_dir = image_dir
		self.image_type = image_type
		self.transform = transform
		self.mode = mode
		self.data_index = []
		images = sorted(glob(self.image_dir+"/images/*.nii"))
		targets = sorted(glob(self.image_dir+"/targets/*.nii"))
		self.data_urls = images
		self.data_targets = targets

		self.data_index = list(range(len(self)))

	def __read(self, url):
		return self.__image_reader[self.image_type](url)

	def __getitem__(self, index):
		img = self.__read(self.data_urls[self.data_index[index]])
		trgt = self.__read(self.data_targets[self.data_index[index]])
		name = os.path.realpath(self.data_urls[self.data_index[index]])

		img = np.squeeze(img)
		# img = img.transpose((1, 0, 2))
		# img -= np.min(img)
		# img /= np.max(img)

		trgt = np.squeeze(trgt)
		# trgt = trgt.transpose((1, 0, 2))
		# trgt -= np.min(trgt)
		# trgt /= np.max(trgt)

		return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(trgt).unsqueeze(0), name

	def __len__(self):
		return len(self.data_urls)

	def __read(self, url):
		return self.__image_reader[self.image_type](url)


def get_loader(image_dir, crop_size=101, image_size=101, 
               batch_size=1, dataset='OASIS_CAPIIO', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []

    if mode == 'train':
    	transform.append(T.RandGaussianNoise())
    	transform.append(T.RandShiftIntensity(30))
    	transform.append(T.RandStdShiftIntensity(3))
    	transform.append(T.RandBiasField())
    	transform.append(T.RandScaleIntensity(0.25))
    	transform.append(T.RandAdjustContrast())
    	transform.append(T.RandGaussianSmooth())
    	transform.append(T.RandGaussianSharpen())
    	transform.append(T.RandHistogramShift())
    	# transform.append(T.RandGibbsNoise())
    	# transform.append(T.RandKSpaceSpikeNoise())
    	transform.append(T.RandRotate())
    	transform.append(T.RandFlip())

    
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = DataFolder(image_dir, 'nii', transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
	loader = get_loader('/home/jgshah1/3dtranslation/fold1/train/', mode="train")
	for i, x in enumerate(loader):
		print(i, x[0].shape, x[1].shape, x[2][0], torch.min(x[0]), torch.max(x[0]))
		break