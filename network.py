import torch, os, sys, cv2, json, argparse, random
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 


def init_weights(m):
	if type(m) != nn.CELU and type(m) != nn.Sequential and type(m) != nn.Upsample and type(m) != nn.ReLU and type(m) != nn.Sigmoid \
		and type(m) != nn.LeakyReLU and type(m) != nn.MaxPool2d and type(m) != nn.Tanh and type(m) != nn.BatchNorm2d:
		nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)


class ShaderballData(Dataset):

	def __init__(self, data_dir, gt_dir):
		super(ShaderballData, self).__init__()

		self.data_dir = data_dir
		self.gt_dir = gt_dir

		self.u_images = os.listdir('%s/uniform/%s/' % (data_dir, gt_dir))
		self.sv_images = os.listdir('%s/spatially_varying/%s/' % (data_dir, gt_dir))

		self.u_len = len(self.u_images)
		self.sv_len = len(self.sv_images)

		self.u_metadata = json.load(open('%s/uniform/metadata.json' % data_dir))
		self.sv_metadata = json.load(open('%s/spatially_varying/metadata.json' % data_dir))

	def __len__(self):
		return self.u_len+self.sv_len

	def __getitem__(self, index):
		if index < self.u_len:
			img_name_og = self.u_images[index]
			metadata = self.u_metadata
			int_dir = 'uniform'
		elif index >= self.u_len:
			img_name_og = self.sv_images[index-self.u_len]
			metadata = self.sv_metadata
			int_dir = 'spatially_varying'

		img_name = img_name_og.split('.')[0]
		img_name = img_name.split('_')
		main_index = img_name[0]
		light_index = int(img_name[1])
		texture_name = metadata[main_index]['texture_name']

		gt = cv2.imread('%s/%s/%s/%s' % (self.data_dir, int_dir, self.gt_dir, img_name_og))
		gt = cv2.resize(gt, (400, 400))
		gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
		gt = gt.astype(np.float) / 255.0
		gt = torch.from_numpy(gt)
		gt = gt.permute((2, 0, 1))

		diffuse = cv2.imread('%s/%s/diffuse/%s' % (self.data_dir, int_dir, texture_name))
		diffuse = cv2.resize(diffuse, (400, 400))
		diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB)
		diffuse = diffuse.astype(np.float) / 255.0
		diffuse = torch.from_numpy(diffuse)
		diffuse = diffuse.permute((2, 0, 1))

		specularity = cv2.imread('%s/%s/specularity/%s' % (self.data_dir, int_dir, texture_name))
		specularity = cv2.resize(specularity, (400, 400))
		specularity = cv2.cvtColor(specularity, cv2.COLOR_BGR2RGB)
		specularity = specularity.astype(np.float) / 255.0
		specularity = torch.from_numpy(specularity)
		specularity = specularity.permute((2, 0, 1))

		normal = cv2.imread('%s/%s/normal/%s' % (self.data_dir, int_dir, texture_name))
		normal = cv2.resize(normal, (400, 400))
		normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
		normal = normal.astype(np.float) / 255.0
		normal = torch.from_numpy(normal)
		normal = normal.permute((2, 0, 1))

		roughness = cv2.imread('%s/%s/roughness/%s' % (self.data_dir, int_dir, texture_name))
		roughness = cv2.resize(roughness, (400, 400))
		roughness = cv2.cvtColor(roughness, cv2.COLOR_BGR2RGB)
		roughness = roughness.astype(np.float) / 255.0
		roughness = torch.from_numpy(roughness)
		roughness = roughness.permute((2, 0, 1))

		sun = list(metadata[main_index]['light_data'][light_index]['sun_direction'])
		sun.append(float(metadata[main_index]['light_data'][light_index]['turbidity']))
		sun = torch.tensor(sun)

		return {
			'name' : img_name_og,
			'sun' : sun.type(torch.float).to('cuda:0'),
			'diffuse' : diffuse.type(torch.float).to('cuda:0'),
			'specularity' : specularity.type(torch.float).to('cuda:0'),
			'normal' : normal.type(torch.float).to('cuda:0'),
			'roughness' : roughness.type(torch.float).to('cuda:0'),
			'gt' : gt.type(torch.float).to('cuda:0')
		}


class NeuralRenderer(nn.Module):

	def __init__(self):
		super(NeuralRenderer, self).__init__()

		self.direction_fc = nn.Sequential(
				nn.Linear(4, 32),
				nn.Tanh(),
				nn.Linear(32, 128),
				nn.Tanh(),
				nn.Linear(128, 256),
				nn.Tanh(),
				nn.Linear(256, 625),
				nn.Tanh()
			)

		# self.shader_conv = nn.Sequential(
		# 		nn.Conv2d(12, 32, 3, padding=1),
		# 		nn.ReLU(),
		# 		nn.Conv2d(32, 64, 3, padding=1, stride=2),
		# 		nn.ReLU(),
		# 		nn.Conv2d(64, 128, 3, padding=1, stride=2),
		# 		nn.ReLU(),
				
		# 		nn.Upsample(scale_factor=2, mode='bilinear'),
		# 		nn.Conv2d(128, 64, 3, padding=1),
		# 		nn.ReLU(),
		# 		nn.Upsample(scale_factor=2, mode='bilinear'),
		# 		nn.Conv2d(64, 64, 3, padding=1),
		# 		nn.ReLU(),
		# 		nn.Conv2d(64, 64, 3, padding=1),
		# 		nn.ReLU()
		# 	)

		# Add shader parameters, along channels
		self.d1 = nn.Sequential(
				nn.Conv2d(12, 64, 3, padding=1, stride=2),
				nn.BatchNorm2d(64),
				nn.ReLU()
			)
		self.d2 = nn.Sequential(
				nn.Conv2d(64, 128, 3, padding=1, stride=2),
				nn.BatchNorm2d(128),
				nn.ReLU()
			)
		self.d3 = nn.Sequential(
				nn.Conv2d(128, 256, 3, padding=1, stride=2),
				nn.BatchNorm2d(256),
				nn.ReLU()
			)
		self.d4 = nn.Sequential(
				nn.Conv2d(256, 512, 3, padding=1, stride=2),
				nn.BatchNorm2d(512),
				nn.ReLU()
			)

		# Add light direction, 25x25 repeated 128 along channels
		self.bottleneck = nn.Sequential(
				nn.Conv2d(640, 512, 3, padding=1),
				nn.BatchNorm2d(512),
				nn.ReLU()
			)

		self.u4 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear'),
				nn.Conv2d(1024, 512, 3, padding=1),
				nn.BatchNorm2d(512),
				nn.ReLU()
			)

		self.u3 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear'),
				nn.Conv2d(768, 256, 3, padding=1),
				nn.BatchNorm2d(256),
				nn.ReLU()
			)
		self.u2 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear'),
				nn.Conv2d(384, 128, 3, padding=1),
				nn.BatchNorm2d(128),
				nn.ReLU()
			)
		self.u1 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear'),
				nn.Conv2d(192, 64, 3, padding=1),
				nn.BatchNorm2d(64),
				nn.ReLU()
			)

		self.final = nn.Sequential(
				nn.Conv2d(64, 3, 3, padding=1),
				nn.Sigmoid()
			)
		
	def initialise_weights(self):
		self.direction_fc.apply(init_weights)

		self.d1.apply(init_weights)
		self.d2.apply(init_weights)
		self.d3.apply(init_weights)
		self.d4.apply(init_weights)
		self.bottleneck.apply(init_weights)
		self.u1.apply(init_weights)
		self.u2.apply(init_weights)
		self.u3.apply(init_weights)
		self.u4.apply(init_weights)

		self.final.apply(init_weights)

	def forward(self, diffuse, specularity, normal, roughness, light_direction):
		shader_inp = torch.cat((diffuse, specularity), dim=1)
		shader_inp = torch.cat((shader_inp, normal), dim=1)
		shader_inp = torch.cat((shader_inp, roughness), dim=1)

		direction_op = self.direction_fc(light_direction)
		direction_op = direction_op.repeat(1, 128)
		direction_op = direction_op.view(-1, 128, 25, 25)

		d1 = self.d1(shader_inp)
		d2 = self.d2(d1)
		d3 = self.d3(d2)
		d4 = self.d4(d3)

		b = self.bottleneck(torch.cat((d4, direction_op), dim=1))

		u4 = self.u4(torch.cat((b, d4), dim=1))
		u3 = self.u3(torch.cat((u4, d3), dim=1))
		u2 = self.u2(torch.cat((u3, d2), dim=1))
		u1 = self.u1(torch.cat((u2, d1), dim=1))

		final = self.final(u1)

		return final








