import torch as trc, cv2

# CLASSIFICATION NET

class ClsNet(trc.nn.Module):
	def __init__(self):
		super().__init__()

		self.conv3_1 = trc.nn.Conv2d(1, 2, 3, padding=1)
		self.conv3_2 = trc.nn.Conv2d(2, 4, 3, padding=1)
		self.conv3_3 = trc.nn.Conv2d(4, 8, 3, padding=1)
		self.conv3_4 = trc.nn.Conv2d(8, 16, 3, padding=1)
		self.conv3_5 = trc.nn.Conv2d(16, 32, 3, padding=1)
		
		self.pool = trc.nn.MaxPool2d(2)

		self.fc_1 = trc.nn.Linear(32*8*8, 256)
		self.fc_2 = trc.nn.Linear(256, 128)
		self.fc_3 = trc.nn.Linear(128, 64)
		self.fc_4 = trc.nn.Linear(64, 32)
		self.fc_5 = trc.nn.Linear(32, 1)

		self.relu = trc.nn.ReLU()
		self.sig = trc.nn.Sigmoid()
	
	def forward(self, x):
		x = self.relu(self.pool(self.conv3_1(x)))
		x = self.relu(self.pool(self.conv3_2(x)))
		x = self.relu(self.pool(self.conv3_3(x)))
		x = self.relu(self.pool(self.conv3_4(x)))
		x = self.relu(self.pool(self.conv3_5(x)))

		x = x.reshape(x.shape[0], -1)

		x = self.relu(self.fc_1(x))
		x = self.relu(self.fc_2(x))
		x = self.relu(self.fc_3(x))
		x = self.relu(self.fc_4(x))

		return self.sig(self.fc_5(x)).squeeze()

# SEGMENTATION NET

class ConvBlock(trc.nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv3_1 = trc.nn.Conv2d(in_ch, out_ch, 3, padding=1)
		self.conv3_2 = trc.nn.Conv2d(out_ch, out_ch, 3, padding=1)
		self.act = trc.nn.ReLU()
	
	def forward(self, x):
		x = self.act(self.conv3_1(x))
		return self.act(self.conv3_2(x))

class Encoder(trc.nn.Module):
	def __init__(self, ch=(3, 16, 32, 64)):
		super().__init__()
		blocks = [ConvBlock(ch[i], ch[i+1]) for i in range(len(ch)-1)]
		self.encBlocks = trc.nn.ModuleList(blocks)
		self.pool = trc.nn.MaxPool2d(2)
		self.act = trc.nn.ReLU()
	
	def forward(self, x):
		blockOutputs = []
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs

class Decoder(trc.nn.Module):
	def __init__(self, ch=(64, 32, 16)):
		super().__init__()
		upsamplers, decBlocks = [], []
		for i in range(len(ch)-1):
			upsamplers.append(trc.nn.ConvTranspose2d(ch[i], ch[i+1], 2, 2))
			decBlocks.append(ConvBlock(ch[i], ch[i+1]))
		
		self.ch = ch
		self.upsamplers = trc.nn.ModuleList(upsamplers)
		self.decBlocks = trc.nn.ModuleList(decBlocks)
	
	def forward(self, x, encFeatures):
		for i in range(len(self.ch) - 1):
			x = self.upsamplers[i](x)

			# crop the current features from the encoder blocks (if needed),
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current decoder block
			# encFeatures_for_cat = self.crop(encFeatures[i], x)
			x = trc.cat([x, encFeatures[i]], dim=1)
			x = self.decBlocks[i](x)
		return x
	
	def crop(self, encFeatures, x):
		_, _, h, w = x.shape
		res = transforms.CenterCrop([h, w])(encFeatures)
		return res

class SegNet(trc.nn.Module):
	def __init__(self, encCH=(1, 16, 32, 64), decCH=(64, 32, 16),
							 nClasses=1, retainDim=True, outSize=(256, 256)):
		super().__init__()
		self.retainDim, self.outSize = retainDim, outSize
		self.encoder, self.decoder = Encoder(encCH), Decoder(decCH)
		self.head = trc.nn.Conv2d(decCH[-1], nClasses, 1)
		self.act = trc.nn.Sigmoid()
	
	def forward(self, x):
		encFeatures = self.encoder(x)
		dec_res = self.decoder(encFeatures[-1], encFeatures[::-1][1:])
		mask = self.head(dec_res)
		# resize to match initial size if needed
		# trc.nn.functional.interpolate(mask, self.outSize)
		return self.act(mask)

# DATASET

class processDataset(trc.utils.data.Dataset):
	def __init__(self, imgPaths, transforms=None):
		self.imgPaths, self.transforms = imgPaths, transforms

	def __len__(self):
		return len(self.imgPaths)
	
	def __getitem__(self, idx):
		if self.imgPaths[idx][-4:] == '.tif':
			image = cv2.imread(self.imgPaths[idx], -1)
		else:
			image = cv2.imread(self.imgPaths[idx])
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		imgShape = image.shape

		if self.transforms is not None:
			image = self.transforms(image)

		imgName = self.imgPaths[idx].split('\\')[-1]
		return image, imgName, self.imgPaths[idx], imgShape
