""" 
Implementation of quantization modules for VQ-VAEs.
We use the Quantizer modules to compute Latent Action prototypes.

Some modules adapted from:
  -  https://github.com/CompVis/taming-transformers/
  -  https://github.com/naver/PoseGPT/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_



def L2_efficient(x, y):
    """ Efficient pairwise euclidean distance """
    return (x.pow(2).sum(1, keepdim=True) - 2 * x @ y + y.pow(2).sum(0, keepdim=True))



class EmaCodebookMeter:
    """
    Compute an estimate of centroid usage using an EMA to
    track proportions of codeword usage.
    
    Args:
    -----
    codebook_size: int
        Number of codewords in the codebook.
    ema_alpha: float
        Weight-value used for the exponential moving average
    """

    def __init__(self, codebook_size, ema_alpha=0.05):
        """ Codebook meter initializer """
        if ema_alpha < 0 or ema_alpha > 1.0:
            raise ValueError(f"{ema_alpha = } must be in [0, 1]...")
        self.codebook_size = codebook_size
        self.ema_alpha = ema_alpha
        self.reset()
        return

    def reset(self):
        """ Resetting the bin-count tracking """
        self.bins = nn.Parameter(
                torch.ones(self.codebook_size) / self.codebook_size,
                requires_grad=False
            )
        self.iters = 0
        return

    def bincount(self, val, weights=None):
        """ Computing current distribution of codewords """
        norm = val.shape[0]
        weights = weights.reshape(-1) if weights is not None else None
        count = torch.bincount(
                val.reshape(-1),
                minlength=self.codebook_size,
                weights=weights
            ).detach()
        self.iters += 1
        return count / norm

    def load(self, bins):
        """ Loading bins """
        self.bins = torch.load(bins)

    def update(self, val, weights=None):
        """
        Count usage of each value in the codebook and updated the codeword usages statistics 
        via an exponential moving average
        """
        count = self.bincount(val, weights=weights)
        alpha = max(self.ema_alpha, 1 / (self.iters + 1))
        self.bins = (1. - alpha) * self.bins + alpha * count.to(self.bins.device)
        return

    def get_hist(self):
        """ Fetching the codeword use statistics """
        return self.bins



class EmaVectorQuantizer(nn.Module):
    """
    Wrapper for Vector Quantizer that enforces the codewords to be updated
    via an Exponential Moving Average. 

    Args:
    -----
    num_embs: int
       Number of centroids (i.e. action prototypes) to learn in the codebook
    emb_dim: int
       Dimension of the centroids (i.e. action prototypes)
    nbooks: int
        Number of codebooks to use. Only nbooks=1 is implemented for now.
    ema_alpha: float
        Weight-value used for the exponential moving average
    """

    def __init__(self, num_embs, emb_dim, nbooks=1, balance=False, ema_alpha=0.05):
        """ Module initializer """
        assert ema_alpha >= 0 and ema_alpha <= 1.0, f"{ema_alpha = } must be in [0, 1]"
        if nbooks != 1:
            raise NotImplementedError(f"Only nbooks=1 is implemented, but {nbooks = }")
        super().__init__()
        self.num_embs = num_embs        
        self.emb_dim = emb_dim
        self.ema_alpha = ema_alpha
        self.epsilon = 1e-6
        self.update = True

        self.vq = VectorQuantizer(
                num_embs=num_embs,
                emb_dim=emb_dim,
                nbooks=nbooks,
                balance=balance
            ).requires_grad_(False)
        
        # EMA paramerters
        self.register_buffer('ema_count', torch.zeros(self.num_embs))
        self.register_buffer('ema_weight', torch.empty((self.num_embs, self.emb_dim)))
        self.ema_weight.data.uniform_(-1 / self.num_embs, 1 / self.num_embs)
        return

    def forward(self, z):
        """ Forward pass  """
        B = z.shape[0]
        epsilon = self.epsilon
        z_q, vq_losses, min_encoding_indices = self.vq(z)
        encodings = F.one_hot(min_encoding_indices, num_classes=self.num_embs)
        
        # Using EMA to update the embedding vectors.   
        if self.training and self.update:
            with torch.no_grad():
                z = z.flatten(0, -2)
                encodings = encodings.flatten(0, -2)
                ema_count = self.ema_alpha * self.ema_count + \
                            (1 - self.ema_alpha) * torch.sum(encodings, 0)
                self.ema_count = (ema_count + epsilon) / (B + self.num_embs * epsilon) * B
                dw = torch.matmul(encodings.float().t(), z)
                self.ema_weight = self.ema_alpha * self.get_buffer('ema_weight') + \
                                  (1 - self.ema_alpha) * dw
                # updating values
                ema_weight = self.get_buffer('ema_weight')
                ema_count = self.get_buffer('ema_count').unsqueeze(1)
                self.vq.embeddings['0'].weight.data = ema_weight / (ema_count + 1e-8)

        return z_q, vq_losses, min_encoding_indices

    def get_codebook_entry(self, indices):
        """ Obtaining the codeword according to the provided indices"""
        return self.vq.get_codebook_entry(indices)

    def get_hist(self):
        """ Fetching the codeword distribution for a given Codebook-EMA module """
        return self.ema_count.cpu().detach()

    def get_codewords(self, i='0'):
        """ Fetching the codewords from a given VQ dictionary """
        return self.vq.get_codewords(i=i)
    
    def get_variability(self, z, action_embs=None, action_idxs=None):
        """ Computing variability embeddings """
        v, scores = self.vq.get_variability(
                z=z,
                action_embs=action_embs,
                action_idxs=action_idxs,
            )
        return v, scores



class VectorQuantizer(nn.Module):
    """
    Vector Quantization module
    
    Args:
    -----
    num_embs: int
        Number of embeddings in the codebook
    emb_dim: int
        Dimensiaonlity of the codebook embeddings
    beta: float
        beta multiplier for the commitment loss in the VQ-loss function
    nbooks: int
        Number of codebooks to use, i.e. product quantization.
        Embeddings get split into N chunks, and each chunk is encoded via a distinct codebook
    balance: bool
        For making the codeword usage more uniform
    """

    def __init__(self, num_embs, emb_dim, nbooks=1, balance=False):
        """ Module initializer """
        if nbooks != 1:
            raise NotImplementedError(f"So far we only support 'nbooks' = 1")
        if balance is not False:
            raise NotImplementedError(f"So far we only support 'balance' = False")
        super().__init__()
        print_("Instanciating VectorQuantizer")
        self.num_embs = num_embs
        self.emb_dim = emb_dim
        self.nbooks = nbooks
        self.balance = balance

        assert num_embs % nbooks == 0, "{nbooks = } should be divisible by {num_embs = }..."
        self.num_embs_i = num_embs // nbooks

        # computing the emb_dim for every codebook. emb_dim need not be divisible by nbooks
        chunk_dims = (nbooks - 1) * [emb_dim // nbooks]
        chunk_dims = chunk_dims + [emb_dim - (nbooks - 1) * (emb_dim // nbooks)]  # remaining
        self.chunk_dims = chunk_dims
        self.embeddings = torch.nn.ModuleDict({
                str(i): nn.Embedding(self.num_embs_i, d) for i, d in enumerate(chunk_dims)
            })

        self.trackers = {}
        for i, e in self.embeddings.items():
            e.weight.data.uniform_(-1.0 / self.num_embs_i, 1.0 / self.num_embs_i)
            print_(f"  --> Codebook {i}: {list(e.weight.size())}")
            self.trackers[int(i)] = EmaCodebookMeter(self.num_embs_i)
        return 

    def get_codewords(self, i='0'):
        """ Fetching the codewords from a given VQ dictionary """
        return self.embeddings[i].weight

    def get_hist(self, i='0'):
        """ Fetching the codeword distribution for a given Codebook-EMA module """
        if isinstance(i, str):
            assert i.isnumeric(), f"Tracker Idx {i} must be a number..."
            i = int(i)
        return self.trackers[i].get_hist()

    def get_state(self):
        """ Getting the codeword distribution for each of the codebook-EMA """
        return {i: self.get_hist(i).cpu().data.numpy() for i in self.trackers.keys()}

    def load_state(self, bins):
        """ Restoring the state of the Codebook-EMA modules """
        for i, b in bins.items():
            self.trackers[i].load(b)

    def reset(self, i):
        """ Restarting the Codebook-EMA modules """
        for i in self.trackers.keys():
            self.trackers = EmaCodebookMeter(self.chunk_dims[int(i)])

    def track_assigment(self, emb_ind, i):
        """ Updating the Codebook-EMA 'i' with the embeddings 'emb_dim' """
        self.trackers[i].update(emb_ind)

    def forward_one(self, z, i, weights=None):
        """ 
        Computing the vector quantization using a single codebook
        
        Args:
        -----
        z: torch tensor
            Embeddings to quantize. Shape is (*, emb_dim)
        i: int
            Index of the codebook to use.
        """
        # getting the actual embedding size for the current codebook
        base_dim = self.emb_dim // self.nbooks
        if i < self.nbooks - 1:
            emb_dim = base_dim
        else:
            emb_dim = self.emb_dim - (self.nbooks - 1) * base_dim

        # pairwise distances between embeddigns and codewords
        z_flattened = z.reshape(-1, emb_dim)
        cur_codewords = self.embeddings[str(i)].weight
        dist = L2_efficient(z_flattened, cur_codewords.t())

        if self.balance and weights is not None:  # balancing usage
            wdist = dist * weights.unsqueeze(0)
            dist = -torch.nn.functional.softmax(-wdist, 1)

        # finding best codeword index
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embs_i).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if self.training:
            self.track_assigment(min_encoding_indices.detach(), i)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings[str(i)].weight).view(z.shape)
        z_q_index = min_encoding_indices.reshape(*z.shape[:-1], 1)
        return z_q, z_q_index

    def forward(self, z, **kwargs):
        """ 
        Computing the vector quantization of some embeddings.
        
        Args:
        -----
        z: torch tensor
            Vectors that we want to quantize. Shape is (..., emb_dim)
        """
        assert z.size(-1) == self.emb_dim, f"{z.shape = } must have {self.emb_dim = }"
        
        # vector quantization with each of the codewords
        zs = torch.split(z, z.size(-1) // len(self.embeddings), dim=-1)
        zq_i = [self.forward_one(z, i, weights=self.get_hist(i)) for i, z in enumerate(zs)]
        z_q, min_encoding_indices = [torch.cat([e[i] for e in zq_i], dim=-1) for i in [0, 1]]

        # compute loss for embedding
        quant_loss = torch.mean((z_q.detach() - z) ** 2, dim=-1)
        commit_loss = torch.mean((z_q - z.detach()) ** 2, dim=-1)
        losses = {
            "quant_loss": quant_loss,
            "commit_loss": commit_loss
        }

        # straight through estimation for copying gradients
        z_q = z + (z_q - z).detach()
        return z_q, losses, min_encoding_indices

    def get_codebook_entry(self, indices):
        """
        Sampling codewords from the dictionary
        
        Args:
        ----
        indices: torch tensor
            Tensor with the codeword indices to sample. Shape is (B, N)
            
        Return:
        -------
        z_q: torch tensor
            Codewords corresponding to the input indices. shape is (B, N, emb_dim)
        """
        embds = [self.embeddings["0"](indices.squeeze(-1))]
        zq = torch.cat(embds, dim=-1)
        return zq
 
    def get_variability(self, z, action_embs=None, action_idxs=None):
        """
        Computing a variability latent based on the distance between the embedding
        and the codebook words.
        
        Args:
        -----
        z: torch tensor
            Embeddings used to compute the variability. Shape is (*, emb_dim)
        action_embs: torch tensor or None
            If provided, action embeddings computed during vector quantization.
            This allows us to avoid recomputing some operations.
            
        Returns:
        --------
        v: torch tensor
            Variability latent vector.
            Shape is (*, emb_dim), with * being determined by the shape of input 'z'
        """
        init_shape, emb_dim = z.shape[:-1], z.shape[-1]
        z_flattened = z.reshape(-1, emb_dim)
        
        # latent is difference between sample and closest codeword
        if action_embs is None or action_idxs is None:
            action_embs, action_idxs = self.forward_one(z, i=0, weights=None)
        action_embs = action_embs.reshape(-1, emb_dim)
        v = z_flattened - action_embs
        scores = F.one_hot(action_idxs, num_classes=self.num_embs)

        v = v.reshape(*init_shape, emb_dim)
        return v, scores
    
   
    
#