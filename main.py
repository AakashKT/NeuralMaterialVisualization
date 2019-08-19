import torch, os, sys, cv2, json, argparse, random, math
import torch.nn as nn
from torch.nn import init
import functools, base64
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image
from io import BytesIO
from werkzeug import secure_filename

import torchvision.transforms as transforms
import numpy as np

from network import *
from flask import Flask , render_template, request, send_file

cloudiness_mapping = {
	0: 2.0,
	1: 3.0,
	2: 6.0,
	3: 10.0
}

app = Flask(__name__)
model = None

def get_sun_direction(phi, theta):
	phi_rad = phi * math.pi / 180
	theta_rad = theta * math.pi / 180

	x = 3.0 * math.cos(phi_rad) * math.cos(theta_rad)
	y = 3.0 * math.cos(phi_rad) * math.sin(theta_rad)
	z = 3.0 * math.sin(phi_rad)

	return x, y, z

x, y, z = get_sun_direction(100, 100)
final_sun = torch.unsqueeze(torch.tensor([x, y, z, 6.0]), dim=0)
# final_sun = torch.unsqueeze(torch.tensor([x, y, z, 6.0]), dim=0).cuda()

def load_checkpoint(filename):
	chkpoint = torch.load(filename);
	model = NeuralRenderer();
	# model.to('cuda:0');
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

	epoch = chkpoint['epoch'];
	model.load_state_dict(chkpoint['state_dict']);
	optimizer.load_state_dict(chkpoint['optimizer']);

	return model, optimizer, int(epoch);

@app.route('/')
def index_ui():
	return render_template('index.html')

@app.route('/diffuse-upload', methods = ['POST'])
def diffuse_upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save('static/'+secure_filename('diffuse.png'))

		i = cv2.resize(cv2.imread('static/diffuse.png'), (200, 200))
		cv2.imwrite('static/diffuse.png', i)

		return render_template('index.html')

@app.route('/normal-upload', methods = ['POST'])
def normal_upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save('static/'+secure_filename('normal.png'))

		i = cv2.resize(cv2.imread('static/normal.png'), (200, 200))
		cv2.imwrite('static/normal.png', i)

		return render_template('index.html')

@app.route('/specular-upload', methods = ['POST'])
def specular_upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save('static/'+secure_filename('specularity.png'))

		i = cv2.resize(cv2.imread('static/specularity.png'), (200, 200))
		cv2.imwrite('static/specularity.png', i)

		return render_template('index.html')

@app.route('/roughness-upload', methods = ['POST'])
def roughness_upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save('static/'+secure_filename('roughness.png'))

		i = cv2.resize(cv2.imread('static/roughness.png'), (200, 200))
		cv2.imwrite('static/roughness.png', i)

		return render_template('index.html')

@app.route('/render', methods=['POST'])
def map_and_render():
	global final_sun

	# theta = 202
	# phi = 154
	# cloudiness = 10
	# x, y, z = get_sun_direction(phi, theta)
	# final_sun = torch.tensor([x, y, z, cloudiness], dtype=torch.float)
	# final_sun = torch.unsqueeze(final_sun, dim=0).cuda()

	os.system('blender-2.80/blender --enable-autoexec -b uv_map.blend')

	diffuse_img = cv2.cvtColor(cv2.imread('diffuse_uv.png'), cv2.COLOR_BGR2RGB)
	diffuse_img = cv2.resize(diffuse_img, (400, 400))
	diffuse_img = diffuse_img.astype(np.float) / 255.0
	diffuse_img = torch.from_numpy(diffuse_img)
	diffuse_img = diffuse_img.permute((2, 0, 1))
	diffuse_img = torch.unsqueeze(diffuse_img, dim=0).type(torch.float)
	# diffuse_img = torch.unsqueeze(diffuse_img, dim=0).type(torch.float).cuda()

	normal_img = cv2.cvtColor(cv2.imread('normal_uv.png'), cv2.COLOR_BGR2RGB)
	normal_img = cv2.resize(normal_img, (400, 400))
	normal_img = normal_img.astype(np.float) / 255.0
	normal_img = torch.from_numpy(normal_img)
	normal_img = normal_img.permute((2, 0, 1))
	normal_img = torch.unsqueeze(normal_img, dim=0).type(torch.float)
	# normal_img = torch.unsqueeze(normal_img, dim=0).type(torch.float).cuda()

	specular_img = cv2.cvtColor(cv2.imread('specularity_uv.png'), cv2.COLOR_BGR2RGB)
	specular_img = cv2.resize(specular_img, (400, 400))
	specular_img = specular_img.astype(np.float) / 255.0
	specular_img = torch.from_numpy(specular_img)
	specular_img = specular_img.permute((2, 0, 1))
	specular_img = torch.unsqueeze(specular_img, dim=0).type(torch.float)
	# specular_img = torch.unsqueeze(specular_img, dim=0).type(torch.float).cuda()

	roughness_img = cv2.cvtColor(cv2.imread('roughness_uv.png'), cv2.COLOR_BGR2RGB)
	roughness_img = cv2.resize(roughness_img, (400, 400))
	roughness_img = roughness_img.astype(np.float) / 255.0
	roughness_img = torch.from_numpy(roughness_img)
	roughness_img = roughness_img.permute((2, 0, 1))
	roughness_img = torch.unsqueeze(roughness_img, dim=0).type(torch.float)
	# roughness_img = torch.unsqueeze(roughness_img, dim=0).type(torch.float).cuda()

	output = model(diffuse_img, specular_img, normal_img, roughness_img, final_sun)
	output = torch.squeeze(output.detach(), dim=0) * 255.0
	output = output.permute((1, 2, 0))
	output = output.cpu().numpy().astype(np.uint8)
	output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

	cv2.imwrite('static/output.png', output)

	return 'OK'

@app.route('/send-parameters', methods=['POST'])
def receive_parameters():
	global final_sun

	# theta = 135
	# phi = 45
	# cloudiness = 3
	theta = int(request.form['theta'])
	phi = int(request.form['phi'])
	cloudiness = int(request.form['cloudiness'])
	cloudiness = cloudiness_mapping[cloudiness]
	x, y, z = get_sun_direction(phi, theta)
	final_sun = torch.tensor([x, y, z, cloudiness], dtype=torch.float)
	final_sun = torch.unsqueeze(final_sun, dim=0)
	# final_sun = torch.unsqueeze(final_sun, dim=0).cuda()

	print('%s %s' % (str(theta), str(phi)))
	print(final_sun)

	diffuse_img = request.form['diffuse_img'].replace('data:image/png;base64,', '')
	diffuse_img = Image.open(BytesIO(base64.b64decode(diffuse_img)))
	diffuse_img = diffuse_img.resize((400, 400), Image.ANTIALIAS)
	diffuse_img.save('static/diffuse.png', 'png')

	normal_img = request.form['normal_img'].replace('data:image/png;base64,', '')
	normal_img = Image.open(BytesIO(base64.b64decode(normal_img)))
	normal_img = normal_img.resize((400, 400), Image.ANTIALIAS)
	normal_img.save('static/normal.png', 'png')

	specular_img = request.form['specular_img'].replace('data:image/png;base64,', '')
	specular_img = Image.open(BytesIO(base64.b64decode(specular_img)))
	specular_img = specular_img.resize((400, 400), Image.ANTIALIAS)
	specular_img.save('static/specularity.png', 'png')

	roughness_img = request.form['roughness_img'].replace('data:image/png;base64,', '')
	roughness_img = Image.open(BytesIO(base64.b64decode(roughness_img)))
	roughness_img = roughness_img.resize((400, 400), Image.ANTIALIAS)
	roughness_img.save('static/roughness.png', 'png')

	map_and_render()

	return 'OK'


if __name__ == '__main__':
	model, _, _ = load_checkpoint('./network_weights.pt')
	model.eval()

	app.run(host='0.0.0.0')
