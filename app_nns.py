import torch as trc, cv2

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

		if self.transforms is not None:
			image = self.transforms(image)

		imgName = self.imgPaths[idx].split('\\')[-1]
		return image, imgName, self.imgPaths[idx]
