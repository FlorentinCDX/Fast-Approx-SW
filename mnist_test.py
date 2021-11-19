import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from sw_approx import sw_approx
import ot
import time
import os

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
data = datasets.MNIST(root=root, train=False, transform=trans, download=True)

n = 10 # number of samples
data_loader = torch.utils.data.DataLoader(
                 dataset=data,
                 batch_size=n,
                 shuffle=False)

samples, _ = next(iter(data_loader))

img_samples = torch.flatten(samples, start_dim=1)

mu_distrib = MultivariateNormal(torch.ones((img_samples.size(1),)), torch.eye(img_samples.size(1)) )
mu_samples = mu_distrib.rsample(sample_shape=torch.Size([n]))

assert img_samples.size() == mu_samples.size()

# Different flavours of Sliced Wasserstein 
slices = [100, 200, 500]
a, b = torch.ones((n,)) / n, torch.ones((n,)) / n
sw = []
t = []
d = ot.sliced.sliced_wasserstein_distance(mu_samples, img_samples, a, b, 1) # To force the interpreter to import the material (avoid lazy computing comportement) 
for n_slices in slices:
    start = time.time()
    d = ot.sliced.sliced_wasserstein_distance(mu_samples, img_samples, a, b, n_slices)
    t.append(time.time() - start)
    sw.append(d)

for i in range(len(slices)):
    print(f"SW with {slices[i]} slices : {sw[i]} ----- time : {t[i]} ")

# Approximation of the Sliced Wasserstein 
start = time.time()
sw_ap = sw_approx(mu_samples, img_samples)
print(f"Approx SW : {sw_ap} ----- time : {time.time() - start}")

