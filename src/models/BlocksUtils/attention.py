"""
Attention modules:
"""

import torch
import torch.nn as nn

from models.BlocksUtils.model_blocks import MLP


__all__ = [
    "SlotAttention",
    "MultiHeadSelfAttention", "MultiHeadCrossAttention",
    "TransformerEncoderBlock", "TransformerDecoder"
]


##################
# SLOT ATTENTION #
##################


class SlotAttention(nn.Module):
    """
    Implementation of the Slot-Attention module from:
      --> Locatello, et al. "Object-centric learning with slot attention." NeurIPS 2020

    Args:
    -----
    dim_feats: integer
        Dimensionality of the input embeddings
    dim_slots: integer
        Dimensionality of the object slots
    Num_slots: integer
        Number of slots competing for representing the image
    num_iters_first: integer
        Number of recurrent iterations to refine the slots for the first video frame.
    num_iters: integer
        Number of recurrent iterations to refine the slots from the second frame onwards.
    mlp_hidden_size: integer
        Hidden dimensionality of the mlp,
    epsilon: float
        Small value used to stabilize divisiona and softmax
    """

    def __init__(self, dim_feats, dim_slots, num_slots, num_iters_first=2, num_iters=2,
                 mlp_hidden=128, epsilon=1e-8):
        """ Module Initializer """
        super().__init__()
        self.dim_slots = dim_slots
        self.num_iters_first = num_iters_first
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.scale = dim_slots ** -0.5

        # normalization layers
        self.norm_input = nn.LayerNorm(dim_feats)
        self.norm_slot = nn.LayerNorm(dim_slots)
        self.norm_mlp = nn.LayerNorm(dim_slots)

        # attention embedders 
        self.to_q = nn.Linear(dim_slots, dim_slots, bias=False)
        self.to_k = nn.Linear(dim_feats, dim_slots, bias=False)
        self.to_v = nn.Linear(dim_feats, dim_slots, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(dim_slots, dim_slots)
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim_slots),
        )
        return

    def forward(self, inputs, slots, step=0, **kwargs):
        """
        Forward pass as depicted in Algorithm 1 from paper

        Args:
        -----
        inputs: torch Tensor
            input feature vectors extracted by the encoder.
            Shape is (Batch, Num locations, Dimensionality)

        Returns:
        --------
        slots: torch Tensor
            Slot assignment for each of the input vectors
            Shape is (Batch, Num Slots, Slot Dimensionality)
        """
        B = inputs.shape[0]
        self.attention_masks = None

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # iterative refinement of the slot representation
        num_iters = self.num_iters_first if step == 0 else self.num_iters
        for _ in range(num_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slot(slots))

            # q ~ (B, N_s, D)
            # k, v ~ (B, N_l, D)
            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale  # ~ (B, N_s, N_l)
            attn = dots.softmax(dim=1) + self.epsilon  # enforcing competition between slots
            self.attention_masks = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)  # ~ (B, N_s, N_l)
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)  # ~ (B, N_s, D)
            # further refinement
            slots = self.gru(
                updates.reshape(-1, self.dim_slots),
                slots_prev.reshape(-1, self.dim_slots)
            )
            slots = slots.reshape(B, -1, self.dim_slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self, shape=None):
        """
        Fetching last computer attention masks

        Returns:
        --------
        attention_masks: torch Tensor
            attention masks highligtinh the importance of each location to each slot
            Shape is (B, N_slots, N_locs)
        """
        B, N_slots, _ = self.attention_masks.shape
        masks = self.attention_masks
        if shape is not None:
            masks = masks.reshape(B, N_slots, *shape)
        return masks



########################
# ATTENTION MECHANISMS #
########################


class MetaAttention(nn.Module):
    """
    MetaClass for (Multi-Head) Key-Value Attention Mechanisms

    Args:
    -----
    emb_dim: integer
        Dimensionality of the input token embeddings.
    head_dim: integer
        Per-head Dimensionality of the tokens.
        Inner attention dimension will be (head_dim * num_heads)
    num_heads: integer
        Number of heads accross which we compute attention.
        If head-dim is None, Emb_dim / Num_Heads division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    self_attn: bool
        If True, it is self-attention, otherwise cross-attention
    kv_dim: int/None
        Dimensionality of the input features used as key and value.
        Only used in cross-attention.
    project_out: bool
        If True, a linear layer is used to process the output of the attention module.
    """

    def __init__(self, emb_dim, head_dim=None, num_heads=1, dropout=0., self_attn=True,
                 kv_dim=None, project_out=False, **kwargs):
        """ Initializer of the Meta-Attention block """
        assert num_heads >= 1
        super().__init__()

        head_dim = head_dim if head_dim is not None else emb_dim
        inner_dim = num_heads * head_dim
        project_out = not inner_dim == emb_dim or project_out
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim

        # computing query, key, value for all embedding heads
        if self_attn:
            kv_dim = emb_dim
        elif kv_dim is None:
            raise ValueError(f"{kv_dim = } cannot be None in cross-attention mode")
        else:
            pass
        self.q = nn.Linear(emb_dim, inner_dim, bias=False)
        self.k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v = nn.Linear(kv_dim, inner_dim, bias=False)

        # output projection
        self.out_proj = nn.Sequential(
                nn.Linear(inner_dim, emb_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        self.attention_masks = None
        return


    def forward(self, x):
        """
        Required forward function. Not implemented in base class
        """
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")


    def attention(self, query, key, value, mask=None, **kwargs):
        """
        Implementation of the standard normalized key-value attention equation
        
        Args:
        -----
        query: torch Tensor
            Query tokens. Shape is (B * num_heads, num_queries, token_dim)
        key/value: torch Tensor
            Key and value tokens, respectively.
            Shape is (B * num_heads, num_keys, token_dim)
        mask: None or torch Tensor
            If given, mask used to enforce causality
            
        Returns:
        --------
        out: torch Tensor
            Output of the Q-K-V attention module.
            Shape is (B * num_heads, num_queries, token_dim)
        """
        scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale  # Q * K.T / sqrt(d_k)
        
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9)
        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        out = torch.einsum('b i j , b j d -> b i d', attention, value)  # Att * V
        return out


    def get_attention_masks(self, reshape=None):
        """ Fetching last computer attention masks """
        if self.attention_masks is None:
            raise ValueError("Attention masks have not yet been computed...")
        masks = self.attention_masks
        return masks


    def split_into_heads(self, x):
        """ Splitting a vector into multiple heads """
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        y = x.reshape(batch_size * self.num_heads, num_tokens, self.head_dim)
        return y


    def merge_heads(self, x):
        """ Rearranging heads and recovering original shape """
        _, num_tokens, dim_head = x.shape
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head)
        return y



class MultiHeadSelfAttention(MetaAttention):
    """
    Multi-Head dot-product self-attention mechanism.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the input token embeddings.
    head_dim: integer
        Per-head Dimensionality of the tokens.
        Inner attention dimension will be (head_dim * num_heads)
    num_heads: integer
        Number of heads accross which we compute attention.
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    project_out: bool
        If True, a linear layer is used to process the output of the attention module.
    """

    def __init__(self, emb_dim, head_dim, num_heads=8, dropout=0., project_out=False):
        """ Initializer of the attention block """
        super().__init__(
                emb_dim=emb_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                self_attn=True,
                dropout=dropout,
                project_out=project_out
            )
        return

    def forward(self, x, **kwargs):
        """
        Forward pass through multi-head self-attention
        """
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, **kwargs)

        # rearranging heads and recovering shape:
        # (B * Nh, N, Dh) --> (B N, Nh, Dh) --> (B, N, D)
        y = self.merge_heads(vect)
        y = self.out_proj(y)
        return y



class MultiHeadCrossAttention(MetaAttention):
    """
    Multi-Head cross-product attention mechanism, as uses in a Transformer decoder.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the input tokens used as queries.
    head_dim: integer
        Per-head Dimensionality of the tokens.
        Innner attention dimension will be (head_dim * num_heads)
    kv_dim: int/None
        Dimensionality of the input features used as key and value.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, head_dim, kv_dim, num_heads=8, dropout=0.):
        """ Initializer of the attention block """
        super().__init__(
                emb_dim=emb_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                dropout=dropout,
                self_attn=False,
                kv_dim=kv_dim
            )
        return

    def forward(self, enc_embs, query_embs, **kwargs):
        """
        Forward pass through multi-head self-attention
        """
        # linear projections and splitting into heads
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(query_embs), self.k(enc_embs), self.v(enc_embs)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, **kwargs)

        # rearranging heads and recovering original shape
        y = self.merge_heads(vect)
        y = self.out_proj(y)
        return y



######################
# TRANSFORMER BLOCKS #
######################


class TransformerBlock(nn.Module):
    """
    Tranformer block from which TransformerEncoder and TransformerDecoder blocks inherit

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    head_dim: int
        Dimensionality of each of the attention heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    self_attn: bool
        If bool, self-attention is applied. Otherwise, we use cross-attention.
    """

    def __init__(self, embed_dim, head_dim, num_heads, mlp_size, self_attn=True):
        """ Module initializer """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.self_attn = self_attn
        assert num_heads >= 1

        # MLP
        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=mlp_size,
        )
        return

    def forward(self, inputs):
        """ Forward pass through transformer """
        raise NotImplementedError("Base Class does not implement 'forward' function...")

    def get_attention_masks(self, reshape=None):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.get_attention_masks(reshape=reshape)
        return attn_masks



class TransformerEncoderBlock(TransformerBlock):
    """
    Tranformer encoder block.

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    head_dim: int
        Dimensionality of each of the attention heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    self_attn: bool
        If bool, self-attention is applied. Otherwise, we use cross-attention.
    """

    def __init__(self, embed_dim, head_dim=32, num_heads=4, mlp_size=256,
                 self_attn=True, project_out=False):
        """ Module initializer """
        super().__init__(
                embed_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                self_attn=True
            )
        # MHA
        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
                emb_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                project_out=project_out
            )
        return

    def forward(self, inputs, mask=None):
        """ Forward pass through transformer encoder block """
        assert inputs.ndim == 3

        # Self-attention.
        x = self.ln_att(inputs)
        x = self.attn(x, mask=mask)
        y = x + inputs
        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y
        return z



class TransformerDecoder(TransformerBlock):
    """
    Transformer decoder block that can cascade both self- and cross-attention
    
    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    head_dim: int
        Dimensionality of each of the attention heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    use_cross_attn: bool
        If True, self-att and cross-attn are both used.
        Otherwise, only self-attention is used.
    kv_dim: int
        Dimensionality of the keys and values in the attention mechanism.
        Only needed if use_cross_attn is True.
    self_attn: bool
        If bool, self-attention is applied. Otherwise, we use cross-attention.
    """
    
    def __init__(self, embed_dim, head_dim, num_heads, mlp_size, kv_dim=None, dropout=0,
                 use_cross_attn=False, project_out=False):
        """ Transformer decoder block that uses both self- and cross-attention """
        super().__init__(
                embed_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                self_attn=False
            )
        self.use_cross_attn = use_cross_attn
        
        # Self-Attention
        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
                emb_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                dropout=dropout,
                project_out=project_out
            )
        
        # Cross Attention
        if use_cross_attn:
            if kv_dim is None:
                raise ValueError(f"If {use_cross_attn = }, 'kv_dim' must be provided...")
            self.ln_cross_att_q = nn.LayerNorm(embed_dim, eps=1e-6)
            self.ln_cross_att_kv = nn.LayerNorm(kv_dim, eps=1e-6)
            self.cross_attn = MultiHeadCrossAttention(
                    emb_dim=embed_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    kv_dim=kv_dim,
                    dropout=dropout
                )
        return
        
    def forward(self, queries, feats=None, self_attn_mask=None, cross_attn_mask=None):
        """ Forward pass """
        assert queries.ndim == 3
        B, L, _ = queries.shape

        # Self-attention.
        x = self.ln_att(queries)
        x = self.attn(x, mask=self_attn_mask)
        y = x + queries

        # Cross-attention.
        if self.use_cross_attn:
            assert feats is not None, f"If {self.use_cross_attn = }, 'feats' must be provided"
            query_embs = self.ln_cross_att_q(y)
            feats = self.ln_cross_att_kv(feats)
            z = self.cross_attn(feats, query_embs=query_embs, mask=cross_attn_mask)
            z = z + y
        else:
            z = y

        # MLP
        out = self.ln_mlp(z)
        out = self.mlp(out)
        out = out + z

        return out


