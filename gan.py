from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from core.descriminator import D
from core.generator import G

#takes as parameter a neural network and defines its weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

print("""\n
\t+-+-+-+-+-+-+-+-+ +-+-+-+
\t|C|R|I|M|I|N|A|L| |G|A|N|
\t+-+-+-+-+-+-+-+-+ +-+-+-+
\tCreated by Davidson Mizael
\t""")

print("\n")
print("# Starting...")

#sets the size of the image (64x64) and the batch (size of the matrix)
batchSize = 64
imageSize = 64

#creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

print("# Loading data...")
dataset = dset.ImageFolder(root = './data', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)


print("# Starting generator and descriminator...")
netG = G()
netG.apply(weights_init)

netD = D()
netD.apply(weights_init)

#training the DCGANs
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

epochs = 15
print("# Starting epochs (%d)..." % epochs)
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        
        #updates the weights of the discriminator nn
        netD.zero_grad()
        
        #trains the discriminator with a real image
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        #trains the discriminator with a fake image
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        ouput = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        #backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        #updates the weights of the generator nn
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG  = criterion(output, target)
        
        #backpropagating the error
        errG.backward()
        optimizerG.step()

        if i == (len(dataloader) - 1):
            print("# Progress: [%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" % (epoch, epochs, i, (len(dataloader) - 1), errD.data[0], errG.data[0]))
            vutils.save_image(real, "%s/real_samples.png" % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results", epoch), normalize = True)

print ("# Finished.")