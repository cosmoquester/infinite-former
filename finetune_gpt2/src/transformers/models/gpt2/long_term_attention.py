# coding: utf-8
"""
Attention modules
"""

import torch
import torch.nn as nn
import torch.distributions as dist

from .basis_functions import GaussianBasisFunctions
from .continuous_sparsemax import ContinuousSparsemax
from .continuous_softmax import ContinuousSoftmax
from typing import Literal

class GaussianBasisFunctionsModule(nn.Module, GaussianBasisFunctions):
    def __init__(self, mu, sigma) -> None:
        super(GaussianBasisFunctionsModule, self).__init__()

        self.register_buffer("mu", mu.unsqueeze(0))
        self.register_buffer("sigma", sigma.unsqueeze(0))

class LongTermAttention(nn.Module):
    def __init__(self, head_dim: int, memory_length: int, attn_func: Literal["softmax", "sparsemax"], 
                  attn_num_basis: int, attn_drop: float, use_infinite_memory: bool, n_heads: int, 
                  use_affines: bool, use_mask: bool, mask_type: Literal["affine", "cnn"],
                  mask_dropout: float, use_kl_regularizer: bool, sigma_0: float, mu_0: float, 
                  use_sticky_memories: bool, **kwargs):
        """
        :param head_dim: hidden dimension of each head
        :param memory_length: memory length
        :param attn_func: attention function softmax or sparsemax
        :param attn_num_basis: the number of long term attention basis
        :param attn_drop: attention dropout
        :param use_infinite_memory: use infinity memory if true
        :param n_heads: the number of heads
        :param use_affines: use affine transform
        :param use_mask: use masking
        :param mask_type: masking type
        :param mask_dropout: masking dropout
        :param use_kl_regularizer: use kl regularizer
        :param sigma_0:
        :param mu_0:
        :param use_sticky_memories: use sticky memory
        """
        super(LongTermAttention, self).__init__(**kwargs)

        self.memory_length = memory_length
        self.head_dim = head_dim
        self.attn_num_basis = attn_num_basis
        self.attn_func = attn_func
        self.n_heads = n_heads
        
        self.use_kl_regularizer = use_kl_regularizer
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0

        self.use_affines=use_affines # whether mu, sigma should be computed using affine transformations

        self.use_mask=use_mask
        self.mask_type=mask_type
        self.mask_dropout_value=mask_dropout
        self.use_sticky_memories=use_sticky_memories

        self.attn_past=None # Used with sticky memory
        self.use_infinite_memory = use_infinite_memory
 
        self.proj_query = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        self.proj_key = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        self.proj_value = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)

        self.attn_out = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_drop)

        if self.use_mask:
            if self.mask_type=='affine':
                self.mask_net=nn.Linear(memory_length,memory_length)
            elif self.mask_type=='cnn':
                self.mask_net=torch.nn.Conv1d(n_heads*head_dim, n_heads*head_dim,3,padding=1)

        if self.mask_dropout_value>0:
            self.mask_dropout=nn.Dropout(self.mask_dropout_value)

        self.nb_samples=512 # number of samples used for update
        self.tau = 0.5 #compressing factor

        self.ridge_penalty=0.5 # ridge penalty
        padding = True

        if self.use_affines:
            self.mu = nn.Linear(attn_num_basis, 1, bias=False)
            self.sigma = nn.Linear(attn_num_basis, 1, bias=False)
            self.softplus = torch.nn.Softplus()

        # get basis functions psi
        sigmas = [0.005, 0.01] # basis function sigmas
        if attn_num_basis % len(sigmas):
            attn_num_basis += (len(sigmas) - attn_num_basis % len(sigmas))

        # get positions for memory vectors
        basis_mu, basis_sigma = LongTermAttention.get_gaussian_basis_functions(attn_num_basis, sigmas)
        self.psi = GaussianBasisFunctionsModule(mu=basis_mu, sigma=basis_sigma)
        self.register_buffer("basis_mu", basis_mu)
        self.register_buffer("basis_sigma", basis_sigma)

        # normalizing function
        if attn_func == 'softmax':
            self.transform = ContinuousSoftmax(psi=[self.psi])
        elif attn_func == 'sparsemax':
            self.transform = ContinuousSparsemax(psi=[self.psi])
        else:
            raise ValueError(f"'attn_func' cannot be `{attn_func}`")

        if padding:
            if memory_length % 2:
                shift = 1 / float(memory_length)
                positions = torch.linspace(-.5+shift, 1.5-shift, 2*memory_length-1)
            else:
                shift = 1 / float(2*memory_length)
                positions = torch.linspace(-.5+shift, 1.5-shift, 2*memory_length)
        else:
            shift = 1 / float(2*memory_length)
            positions = torch.linspace(shift, 1-shift, memory_length)

        # compute basis functions
        Gs = self.compute_G(memory_length, self.psi, positions, padding=padding) # [L,N]
        self.register_buffer("Gs", Gs)
        self.positions = positions[int(memory_length/2):-int(memory_length/2)]

        # compute samples for memory update
        if self.use_infinite_memory:
            tm_tau = torch.arange(1,self.nb_samples+1).float()
            tm_l = torch.arange(self.nb_samples+1,memory_length+self.nb_samples+1).float()
            tm_tau = tm_tau*self.tau/self.nb_samples # positions of old vectors
            tm_l = self.tau + (1-self.tau)*(tm_l-self.nb_samples)/memory_length # positions of new vectors
            positions_inf = torch.cat([tm_tau, tm_l],0) # positions

            if padding:
                if memory_length % 2:
                    shift = 1 / float(memory_length+self.nb_samples)
                    positions_pad_ = torch.linspace(-.5+shift, 0, 2*(memory_length+self.nb_samples)-1)
                else:
                    shift = 1 / float(2*memory_length+self.nb_samples)
                    positions_pad = torch.linspace(-.5+shift, 1.5-shift, 2*(memory_length+self.nb_samples))
                positions_pad_ = torch.FloatTensor([i for i in positions_pad if i<0])
                positions_pad__ = torch.FloatTensor([i for i in positions_pad if i>1])
                positions_inf = torch.cat([positions_pad_,positions_inf,positions_pad__], dim=0)

            self.samples=None
            for t in tm_tau:
                if self.samples is None:
                    self.samples = self.psi.evaluate(t/self.tau)
                else:
                    self.samples = torch.cat([self.samples,self.psi.evaluate(t/self.tau)], dim=0)

            # compute G for the infinite case
            G_inf = self.compute_G(self.nb_samples+memory_length, self.psi, positions_inf, padding=padding) #[L+nb_samples,N]
            self.register_buffer("G_inf", G_inf)

            if self.use_sticky_memories:
                self.register_buffer("bins", torch.linspace(0,1,129)) #self.positions
                self.nb_bins_cat=1
                self.bins_cat = dist.Categorical(torch.ones(self.nb_bins_cat))

    @staticmethod
    def get_gaussian_basis_functions(nb_basis: int, sigmas) -> torch.Tensor:
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten()
        sigma = sigma.flatten()
        assert mu.size(0) == nb_basis
        return mu, sigma

    def compute_G(self, l: int, psi: GaussianBasisFunctionsModule, positions: torch.Tensor, padding=True) -> torch.Tensor:
        F = torch.zeros(self.attn_num_basis, positions.size(0))

        F[:, :] = psi.evaluate(positions.unsqueeze(1)).t()

        I = torch.eye(self.attn_num_basis)
        G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())

        if padding:
            if l % 2:
                G = G[((l-1)//2):(-(l-1)//2), :]
            else:
                G = G[(l//2):-(l//2), :]

        return G

    def score(self, query, keys):
        query = query/ (self.head_dim ** 0.5) # divide by sqrt(head_dim) [B,h,q,d]
        keys = keys.transpose(-1, -2) #[B,h,d,N]
        scores = torch.matmul(query, keys) #[B,h,q,N] 
        return scores

    def value_function(self, x, inf=False):
        if inf:
            G = self.G_inf # [nb_sample+L,N]
        else:
            G = self.Gs # [L,N]
        B = torch.matmul(x, G) # [B,e,N]
        B = B.permute(0,2,1) # [B,N,e]
        
        return B

    def update_inf(self, x, B_past):
        if B_past is not None:       
            if self.use_sticky_memories:

                n = dist.Normal(self.attn_past[0],self.attn_past[1])
                
                bins = self.bins.clone()
                bins[0]=-.000001
                bins[-1]=1.000001

                p = (n.cdf(bins[1:].repeat(self.attn_past[0].size(1),x.size(0),1).permute(2,1,0))
                    -n.cdf(bins[:-1].repeat(self.attn_past[0].size(1),x.size(0),1).permute(2,1,0))).sum(-1).transpose(1,0)
                
                p = (p/p.sum(-1).repeat(p.size(-1),1).transpose(1,0))
                p = dist.Categorical(p)

                b = p.sample((self.nb_samples,))
                
                t = self.bins_cat.sample((self.nb_samples,self.attn_past[0].size(0))).to(x.device)

                ts = (t*(self.bins[b+1]-self.bins[b])/self.nb_bins_cat +self.bins[b]).transpose(1,0)

                ts = torch.sort(ts,-1)[0]
            
                samples=torch.zeros(x.size(0),self.nb_samples,self.attn_num_basis, device=x.device)
                for i in range(len(ts)):
                    samples[i] = self.psi.batch_evaluate(ts[i])

                xm_tau = B_past.transpose(-1,-2).matmul(samples.transpose(-1,-2)) # [B,e,nb_samples]
            
            else:
                xm_tau = B_past.transpose(-1,-2).matmul(self.samples.transpose(-1,-2)) # [B,e,nb_samples]
            
            x = torch.cat([xm_tau,x], dim=2) # [B,e,nb_samples+L]
            B = self.value_function(x, inf=True) # [B,N,e]
        else:
            B = self.value_function(x)

        return B


    def forward(self, k, query, B_past=None):
        """
        Args:
            k: memory [BatchSize, SeqLength, HiddenDim]
            query: query shaped [BatchSize, NumHeads, SeqLength, HeadDim]
        """
        batch_size = k.size(0) #batch size
        qlen = query.size(2) #query length

        k = k.permute(0,2,1) # [B,e,L]
        if self.mask_dropout_value>0:
            k = self.mask_dropout(k)

        if self.use_mask:
            reg_mask=torch.sigmoid(self.mask_net(k))
            k = k*reg_mask

        # perform memory update
        if self.use_infinite_memory:
            B = self.update_inf(k, B_past)
            new_B_past = B.detach()
        else: # compute input continuous approximation
            B = self.value_function(k) # [B,N,e]
            new_B_past = None

        keys = self.proj_key(B)
        values = self.proj_value(B)

        keys = keys.view(batch_size,self.attn_num_basis,self.n_heads,self.head_dim).transpose(1,2) # [B,h,N,d]
        values = values.view(batch_size,self.attn_num_basis,self.n_heads,self.head_dim).transpose(1,2) # [B,h,N,d]

        #compute scores
        scores = self.score(query, keys) #[B,h,q,N] 

        #compute mu and sigma
        if self.use_affines:
            mu = torch.sigmoid(self.mu(scores)) #[B,h,q] 
            sigma_sq = self.softplus(self.sigma(scores))#[B,h,q] 
            mu = mu.view(-1)
            sigma_sq = torch.clamp(sigma_sq, min=1e-4).view(-1)

            if self.use_sticky_memories:
                self.attn_past=[mu.view(batch_size,-1),sigma_sq.view(batch_size,-1)**(1/2)]
        else:
            scores = torch.softmax(scores,dim=-1)
            mu = torch.matmul(scores, self.basis_mu)
            sigma_sq = torch.matmul(scores, self.basis_mu**2 + self.basis_sigma**2) -mu**2
            mu=mu.view(-1)
            sigma_sq=sigma_sq.view(-1)

        if self.use_kl_regularizer:
            sigma_0_sq = self.sigma_0**2
            if self.mu_0 >0:
                kl_reg = 1/2 * ( sigma_sq.view(batch_size,-1) / sigma_0_sq - 
                            torch.log(sigma_sq.view(batch_size,-1)/sigma_0_sq) -1 )
            else:
                kl_reg = 1/2 * ( sigma_sq.view(batch_size,-1) / sigma_0_sq - 
                            torch.log(sigma_sq.view(batch_size,-1)/sigma_0_sq) -1 +
                            (mu.view(batch_size,-1) - self.mu_0)**2 / sigma_0_sq )
        else:
            kl_reg = None

        # pass parameters to theta
        theta = torch.zeros(batch_size*self.n_heads*qlen, 2, device=k.device)  # [B*h*q, 2]
        theta[:, 0] = mu / sigma_sq
        theta[:, 1] = -1. / (2. * sigma_sq)

        #compute basis functions expectation
        r = self.transform(theta) # [B*h*q,N] 

        r = r.view(batch_size,self.n_heads,qlen,self.attn_num_basis).permute(0,1,3,2) # [B,h,N,q]

        values = values.transpose(-1,-2) # [B,h,d,N]

        context = torch.matmul(values,r) # [B,h,d,q]

        context = context.permute(0,3,1,2) # [q,B,h,d]
        context = context.contiguous().view(batch_size,qlen,self.n_heads*self.head_dim) # [q,B,e]

        context = self.attn_out(context)

        return context, new_B_past, kl_reg

    def __repr__(self):
        return "ContinuousAttention"

