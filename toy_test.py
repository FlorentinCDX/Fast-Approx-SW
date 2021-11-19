import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from sw_approx import sw_approx
import ot
import time
    
m1 = torch.tensor([1., 2., 3.]) 
m2 = torch.tensor([4., 5., 6.]) 
sig1 = torch.tensor([[1., 1., 1.], [1., 2., 2.], [1., 2., 3.]]) 
sig2 = torch.eye(3) 
mu_distrib = MultivariateNormal(m1, sig1)
nu_distrib = MultivariateNormal(m2, sig2)

n = 1000 # number of samples
mu_samples = mu_distrib.rsample(sample_shape=torch.Size([n]))
nu_samples = nu_distrib.rsample(sample_shape=torch.Size([n]))

# True Wasserstein 
w = torch.norm(m1 - m2, p=2) + torch.trace(sig1 + sig2 - 2*torch.sqrt(torch.sqrt(sig1) * sig2 * torch.sqrt(sig1)))
print("true Wasserstein  :", w)

# Different flavours of Sliced Wasserstein 
slices = [10, 50, 100]
a, b = torch.ones((n,)) / n, torch.ones((n,)) / n
sw = []
t = []
d = ot.sliced.sliced_wasserstein_distance(mu_samples, nu_samples, a, b, 1) # To force the interpreter to import the material (avoid lazy computing comportement) 
for n_slices in slices:
    start = time.time()
    d = ot.sliced.sliced_wasserstein_distance(mu_samples, nu_samples, a, b, n_slices)
    t.append(time.time() - start)
    sw.append(d)

for i in range(len(slices)):
    print(f"SW with {slices[i]} slices : {sw[i]} ----- time : {t[i]} ---- approx error {torch.abs(sw[i] - w)}")

# Approximation of the Sliced Wasserstein 
start = time.time()
sw_ap = sw_approx(mu_samples, nu_samples)
print(f"Approx SW : {sw_ap} ----- time : {time.time() - start} ---- approx error {torch.abs(sw_ap - w)}")





