import numpy as np
from PIL import Image
import torch
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import models

def MAE(img1, img2, l1loss):
	return l1loss(img1,img2).item()

def PSNR(img1, img2, mseloss, data_range):
	# you could also use skimage
	# import skimage
	# skimage.metrics.peak_signal_noise_ratio(img1.numpy(), (img2*0.99).numpy(),data_range=data_range)
	return 10*torch.log10( (data_range*data_range) / mseloss(img1,img2) )

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def create_window_3D(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t())
	_3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)
	
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
	mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)

	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)
	


class SSIM(torch.nn.Module):
	def __init__(self, window_size = 11, size_average = True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)
			
			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)
			
			self.window = window
			self.channel = channel


		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
	
	
class SSIM3D(torch.nn.Module):
	def __init__(self, window_size = 11, size_average = True):
		super(SSIM3D, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window_3D(window_size, self.channel)

	def ssim_3D(self, img1, img2, window, window_size, channel, size_average = True):
		mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
		mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

		mu1_sq = mu1.pow(2)
		mu2_sq = mu2.pow(2)

		mu1_mu2 = mu1*mu2

		sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
		sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
		sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

		C1 = 0.01**2
		C2 = 0.03**2

		ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

		if size_average:
			return ssim_map.mean()
		else:
			return ssim_map.mean(1).mean(1).mean(1)
	
	def forward(self, img1, img2):
		(_, channel, _, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window_3D(self.window_size, channel)
			
			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)
			
			self.window = window
			self.channel = channel


		return self.ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


class PerceptualLoss(torch.nn.Module):
    '''
    content loss and style loss extracted by selected model
    '''
    def __init__(self, device, model_type='vgg19', content_layers=['conv_4'],
                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], channel_idx=0):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.model_type = model_type
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.channel_idx = channel_idx   # if output and target have multiple channels
        if model_type == 'vgg19':
            self.model = models.vgg19(pretrained=True).features.to(device)
        self.model.eval()

    def normalize_image(self, image):
        # normalize to 0~1
        image = image / image.max()
        # image /= image.max()      # in-place operation, can't compute gradient

        # grayscale to rgb
        image = image[:, self.channel_idx, :, :]
        image = image.unsqueeze(1).expand(-1, 3, -1, -1)

        # normalize by mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)
        image = (image - mean) / std

        # crop the middle region
        crop_size = (image.shape[-1] - 224) // 2
        image = image[:, :, crop_size:crop_size+224, crop_size:crop_size+224]

        return image

    def content_loss(self, feature_prediction, feature_target):
        content_loss = F.mse_loss(feature_prediction, feature_target)
        return content_loss

    def style_loss(self, feature_prediction, feature_target):
        gram_prediction = self.gram_matrix(feature_prediction)
        gram_target = self.gram_matrix(feature_target)
        style_loss = F.mse_loss(gram_prediction, gram_target)
        return style_loss

    def gram_matrix(self, feature):
        batch_size, num_ch, height, width = feature.size()  # NCHW
        feature = feature.view(batch_size * num_ch, height * width)
        gram = torch.mm(feature, feature.t())
        return gram.div(batch_size * num_ch * height * width)

    def forward(self, prediction, target):
        # noramlize image
        prediction = self.normalize_image(prediction)
        target = self.normalize_image(target)

        # get features from selected layers
        conv_block_idx = 0
        model_new = torch.nn.Sequential().to(self.device)
        content_losses = []
        style_losses = []
        # feature_prediction = prediction
        # feature_target = target
        for i, layer in enumerate(self.model):
            # TODO: might have error when it's self defined model
            # the official code given by tutorial, a bit faster than the code below
            if isinstance(layer, torch.nn.Conv2d):
                conv_block_idx += 1
                name = 'conv_' + str(conv_block_idx)
            else:
                name = str(i)
            model_new.add_module(name, layer)

            # get loss
            if name in self.content_layers or name in self.style_layers:
                feature_prediction = model_new(prediction)
                feature_target = model_new(target)
                if name in self.content_layers:
                    content_losses.append(self.content_loss(feature_prediction, feature_target))
                if name in self.style_layers:
                    style_losses.append(self.style_loss(feature_prediction, feature_target))

            # another version without building the new model
            # feature_prediction = layer(feature_prediction)
            # feature_target = layer(feature_target)
            # if isinstance(layer, nn.Conv2d):
            #     conv_block_idx += 1
            #     name = 'conv_' + str(conv_block_idx)
            #     if name in self.content_layers:
            #         content_losses.append(self.content_loss(feature_prediction, feature_target))

        return content_losses, style_losses


	
def ssim(img1, img2, window_size = 11, size_average = True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)
	
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)
	
	return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
	(_, channel, _, _, _) = img1.size()
	window = create_window_3D(window_size, channel)
	
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)
	
	return _ssim_3D(img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
	# pdb.set_trace()
	origin_img = torch.rand((256,256,256))
	generate_img = torch.clone(origin_img)
	print(origin_img.shape)

	# MAE is L1Loss, 0 is the ideal
	L1Loss=torch.nn.L1Loss()
	print('MAE score: ', MAE(origin_img, generate_img, L1Loss))

	# Peak Signal to Noise Ratio (PSNR), bigger is better
	MSE_loss = torch.nn.MSELoss()
	# normally, 8 bit color image's range should be 255
	data_range = 10 
	print('PSNR score: ', PSNR(origin_img, generate_img*0.99, MSE_loss, data_range))

	# Structural Similarity Metric (SSIM), 1 is the ideal
	# https://github.com/jinh0park/pytorch-ssim-3D
	# input should be shape (batch_size, channel, long, width, height)
	img1 = origin_img.cuda()
	img1 = torch.unsqueeze(img1,0)
	img1 = torch.unsqueeze(img1,0)
	img2 = generate_img.cuda()
	img2 = torch.unsqueeze(img2,0)
	img2 = torch.unsqueeze(img2,0)

	# SSIM function
	print(ssim3D(img1,img2).item())
	# SSIM class (loss)
	ssim_loss = SSIM3D(window_size=11)
	print(ssim_loss(img1,img2))


	pdb.set_trace()