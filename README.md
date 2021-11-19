# Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections

This repository provides a official implementation of the paper [Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections](https://arxiv.org/pdf/2106.15427.pdf) from Kimia Nadjahi, Alain Durmus, Pierre E. Jacob, Roland Badeau, Umut Şimşekli.

## Table of Contents

- [Install](#install)
- [Reproduction](#reproduction)
- [Citation](#citation)

## Installation

Clone this repo.
```bash
git git@github.com:FlorentinCDX/Fast-Approx-SW.git
```

This code requires PyTorch 1.0, POT and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Reproduction

This section provides instructions on how to reproduce results in the original paper.

Here as an example we first sample from two different Gaussian distributions. The Wasserstein as an explicit form for those distributions so we can compute it, the approximate version and check the approximation error.

```python3
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from sw_approx import sw_approx
import time

## First sample from two different distributions 
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

# Approximation of the Sliced Wasserstein 
start = time.time()
sw_ap = sw_approx(mu_samples, nu_samples)
print(f"Approx SW : {sw_ap} ----- time : {time.time() - start} ---- approx error {torch.abs(sw_ap - w)}")
```

## Citation

Once again, this is an unofficial implementation, here is the real paper citation and the POT toolbox :

@misc{nadjahi2021fast,
      title={Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections}, 
      author={Kimia Nadjahi and Alain Durmus and Pierre E. Jacob and Roland Badeau and Umut Şimşekli},
      year={2021},
      eprint={2106.15427},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}

@article{flamary2021pot,
  author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
  title   = {POT: Python Optimal Transport},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {78},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-451.html}
}