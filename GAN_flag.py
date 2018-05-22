import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np
import os
import random
from PIL import Image  
from black_box_detector.read_data import MyDataset
#from black_box_detector.blackbox_cnn import test_for_GAN
from black_box_detector.blackbox_multiperception import test_for_GAN
from interpretor import Interpretor

print('==> Preparing data..')

#Hyper-Parameters--

#setting the learning rate
lr = 0.0001

#Number of samples to take a random distribution from
rand_num = 9

#Number of epochs
num_epochs = 1

#select min batch size
BATCH_SIZE = 32

# decide whether to write checkpoint
whether_checkpoint = True

# decide whether to load checkpoint
need_load_checkpoint = True

# Number of each saving checkpoint
save_interval = 1

# Location of checkpoint save folder
save_folder = './IDSGAN_checkpoint'

# from what loaction to start training
load_epoch = 0

# List of features added noise
feature_as_noise_selection = 'delegation'
print('The way of selecting features added noise: ' + feature_as_noise_selection)

z_dimension = 9


# load dataset in complete form as tensor for networks
train_data = MyDataset(root = './nsl_data_bioclass/pro_unnormalized/1-3_norm/nor_nslkddcup_test_norm.csv', for_who = 'black_box')
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

#Number of batches per Epoch
num_batches = len(train_loader)

batch_size = 12833/(len(train_loader)+1)

# load dataset in usable form as tensor for networks
train_data_with_usable = MyDataset(root = './nsl_data_bioclass/pro_unnormalized/1-3_norm/mal_nslkddcup_test_norm.csv', for_who = 'GAN')
train_with_usable_loader = Data.DataLoader(dataset = train_data_with_usable, batch_size = batch_size, shuffle = True, num_workers = 2)

#Just a CNN with 4 convolution layers and 1 fully connected layers!
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(11*50, 6*256),
			#nn.LeakyReLU()
			nn.ReLU(),
		)

		self.cnntra1 = nn.Sequential(
			nn.ConvTranspose1d(256, 128, 1, stride=2, padding=0, bias=False), # 8x11
			nn.BatchNorm1d(128),
			#nn.LeakyReLU(),
			nn.ReLU(True),
			)
		self.cnntra2 = nn.Sequential(
			nn.ConvTranspose1d(128, 64, 1, stride=2, padding=0, bias=False), # 14x21
			nn.BatchNorm1d(64),
			#nn.LeakyReLU(),
			nn.ReLU(True),
			)
		self.cnntra3 = nn.Sequential(
			nn.ConvTranspose1d(64, 11, 1, stride=2, padding=0, bias=False), # 25x40
			#nn.BatchNorm1d(11),
			#nn.LeakyReLU(),
			nn.Tanh(),
			)

	def forward(self, z):
		x = z.view(-1, 11*50)
		x = self.model(x)
		x = x.view(-1, 256, 6)
		#print(x.size())
		x = self.cnntra1(x)
		#print(x.size())
		x = self.cnntra2(x)
		#print(x.size())
		x = self.cnntra3(x)
		#print(x.size())
		x = x.view(-1, 11, 41)
		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv1d(11, 64, 3, stride=1, padding=1), 
			nn.LeakyReLU(),
			nn.MaxPool1d(2), 
			)
		self.cnn2 = nn.Sequential(
			nn.Conv1d(64, 256, 3, stride=1, padding=1), 
			nn.LeakyReLU(),
			nn.MaxPool1d(2), 
			)
		self.fc = nn.Sequential(
			nn.Linear(256*10, 128),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.cnn1(x)
		#print(x.size())
		x = self.cnn2(x)
		x = x.view(-1, 256*10)
		x = self.fc(x)
		print(x.size())
		return x

class Discriminator1(nn.Module):
	def __init__(self):
		super(Discriminator1, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(41*11, 128),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = x.view(-1, 41*11)
		x = self.fc(x)
		return x

def save_checkpoint(model, foldername, filename):
	print("=> saving checkpoint model to %s" % filename)
	torch.save(model.state_dict(), os.path.join(foldername, filename))

def load_checkpoint(model, foldername, filename):
	model.load_state_dict(torch.load(os.path.join(foldername, filename)))

def Usable_Plus_Noise(step):
	original = 0
	for m, (usable_sequences, _) in enumerate(train_with_usable_loader):
		if m == step:
			original = usable_sequences
			break
	z_batch_size = original.size()[0]
	noise = Variable(torch.randn(z_batch_size, 11, 9)) # noise = Variable(torch.from_numpy(np.random.rand(z_batch_size, 11, 9)))
	z = torch.cat((Variable(original), noise), 2)
	z = generator(z)
	z = supplant(original, z, step)
	return(z, z_batch_size)

def supplant(original, z, step):
	#unchange_location_list = [4, 5, 34, 33, 39, 32, 2, 3, 0,]
	for j in range(original.size()[0]):
		max_list = torch.max(original[j], 0)[1].numpy()
		unchange_location_list = [4, 5, 34, 33, 39, 32, 2, 3, 0,]
			 #11, 40, 35, 23,  1, 22, 31, 28, 36, 9, 29,26, 30, 27, 37, 38, 21, 12, 10, 24, 25,7, 15, 16, 13, 18, 17, 8, 20, 14, 6, 19]
		for i in unchange_location_list:
			print(i)
			z[j, :, i] = 0
			z[j, max_list[i], i] = 1
	return z

discriminator = Discriminator1()
generator = Generator()

# There may be a problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if load_epoch >= 0 and need_load_checkpoint==True:
	print("load the epoch from %d" %load_epoch)
	folder_path = '%s/IDSGAN_epoch_%d' % (save_folder, load_epoch)
	file_path_D = 'IDSGAN_epoch_%d_D.pkl' % load_epoch
	load_checkpoint(discriminator, folder_path, file_path_D)

	file_path_G = 'IDSGAN_epoch_%d_G.pkl' % load_epoch
	load_checkpoint(generator, folder_path, file_path_G)

# The classification loss of Discriminator, binary classification, 1 -> real sample, 0 -> fake sample
criterion = nn.BCELoss()

# Define optimizers
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr*2)

# Draw 9 samples from the input distribution as a fixed test set
# Can follow how the generator output evolves
rand_z = Variable(torch.randn(rand_num, 100))

gen_loss=[]
dis_loss=[]


final_noise = 0

for ep in range(num_epochs):
	acc = []
	for n, (real_sequences, targets) in enumerate(train_loader):
		targets_after_BB, _ = test_for_GAN('./black_box_detector/12_MLP_checkpoint/blackbox_epoch_2.pkl', Variable(real_sequences), targets)
		# Cautions!!! The labels get from "test_for_GAN" is typified as LongTensor
		targets_after_BB = torch.from_numpy(targets_after_BB)
		real_sequences = Variable(real_sequences)
		labels_real = Variable(targets_after_BB)


		# Train the discriminator, it tries to discriminate between real and fake (generated) samples
		
		outputs = discriminator(real_sequences)
		#outputs = torch.max(outputs, 1)[1].view(labels_real.size())

		# Cautions!!! The labels get from blackbox is typified as LongTensor
		#             The labels get from discriminator is typified as FloatTensor
		labels_real = labels_real.type(torch.FloatTensor)
		outputs = outputs.type(torch.FloatTensor)
		loss_real = criterion(outputs, labels_real)

		
		# Train the discriminator using the sequence containing noise
		#z = Variable(torch.randn(batch_size, 100))
		#noise = generator(z)
		usable_plus_noise, z_batch_size = Usable_Plus_Noise(n)
		outputs = discriminator(usable_plus_noise.detach())
		#outputs = torch.max(outputs, 1)[1].view(labels_real.size())
		# The new sequence containing usable and noise is classified by blackbox and gets the label as its label
		labels_fake, _ = test_for_GAN('./black_box_detector/12_MLP_checkpoint/blackbox_epoch_2.pkl', 
						usable_plus_noise, torch.ones(z_batch_size).type(torch.LongTensor))
		labels_fake = Variable(torch.from_numpy(labels_fake))
		labels_fake = labels_fake.type(torch.FloatTensor)
		outputs = outputs.type(torch.FloatTensor)
		loss_fake = criterion(outputs, labels_fake)
		
		optimizer_d.zero_grad()
		loss_d = loss_real + loss_fake			  # Calculate the total loss
		loss_d.backward()						   # Backpropagation
		optimizer_d.step()						  # Update the weights


		# Train the generator, it tries to fool the discriminator
		# Draw samples from the input distribution and pass to generator
		#z = Variable(torch.randn(batch_size, 100))
		#noise = generator(z)
		usable_plus_noise, z_batch_size = Usable_Plus_Noise(n)

		# Pass the genrated images to discriminator
		outputs = discriminator(usable_plus_noise.detach())

		labels_fake, accuracy = test_for_GAN('./black_box_detector/12_MLP_checkpoint/blackbox_epoch_2.pkl', 
							usable_plus_noise, torch.ones(z_batch_size).type(torch.LongTensor))
		acc.append(accuracy)
		# Cautions!!! The labels get from blackbox is typified as LongTensor
		#             The labels get from generator is typified as FloatTensor
		labels_to_normal = Variable(torch.zeros(z_batch_size, 1))
		outputs = outputs.type(torch.FloatTensor)
		optimizer_g.zero_grad()
		loss_g = criterion(outputs, labels_to_normal)   # Calculate the loss
		loss_g.backward()                          # Backpropagation
		optimizer_g.step()                         # Update the weights

		final_noise = usable_plus_noise		


		if not n%300 and n != 0:
			print('epoch: {} - loss_d: {} - loss_g: {}'.format(ep, loss_d.data[0], loss_g.data[0]))

	# Save the test results after each Epoch
	print('Iter-{}; D_loss: {}; G_loss: {}'.format(ep, loss_d.data.cpu().numpy(), loss_g.data.numpy()))
	dis_loss.append(loss_d.data.numpy())
	gen_loss.append(loss_g.data.numpy())

	print('acc: %.4f' % (float(sum(acc)/len(acc))))

	# save checkpoint
	if float(sum(acc)/len(acc))<5.0:
		if whether_checkpoint and ep % save_interval == 0:
			folder_path = '%s/IDSGAN_epoch_%d_%f' % (save_folder, ep, float(sum(acc)/len(acc)))
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)

			file_path_D = 'IDSGAN_epoch_%d_D.pkl' % ep
			save_checkpoint(discriminator, folder_path, file_path_D)

			file_path_G = 'IDSGAN_epoch_%d_G.pkl' % ep
			save_checkpoint(generator, folder_path, file_path_G)

# write down the final sequence containing noise
#Interpretor(final_noise)
