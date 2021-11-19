import torch 

def cov(m, y=None):
    '''
    Estimate a covariance matrix given data
    '''
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

def m_2(X):
    return torch.mean(torch.pow(X, 2), dim=0)


def sw_approx(mu: torch.Tensor ,nu: torch.Tensor) -> float:    
    '''
    """
    Central Limite Theorem approximation of the Sliced Wasserstein distance

    .. math::
        \widehat{\mathbf{S W}}_{2}^{2}\left(\mu_{d}, \nu_{d}\right)=\mathbf{W}_{2}^{2}\left\{\mathrm{~N}\left(0, \mathrm{~m}_{2}\left(\bar{\mu}_{d}\right)\right), \mathrm{N}\left(0, \mathrm{~m}_{2}\left(\bar{\nu}_{d}\right)\right)\right\}+(1 / d)\left\|\mathbf{m}_{\mu_{d}}-\mathbf{m}_{\nu_{d}}\right\|^{2}


    Parameters
    ----------
    mu : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    nu : ndarray, shape (n_samples_b, dim)
        samples in the target domain

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    '''
    m_mu = torch.mean(mu, dim=0)
    m_nu = torch.mean(nu, dim=0)
    ### First lets compute d:=W2{N(0, m2(µd_bar)), N(0, m2(νd_bar))}
    # Centered version of mu and nu
    mu_bar = mu - m_mu
    nu_bar = nu - m_nu
    # Compute Wasserstein beetween two centered gaussians
    W = torch.pow(torch.sqrt(m_2(mu_bar)) - torch.sqrt(m_2(nu_bar)), 2)
    
    ## Compute the mean residuals
    d = mu.size(1)
    res = (1/d) * torch.pow(m_mu - m_nu, 2)
    
    ## Approximation of the Sliced Wasserstein 
    return torch.norm(W + res, p=2)
    
    