"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# Seed
def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### Top-k and nucleus sampling ops
def top_k_top_p_filtering(probs, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert probs.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, probs.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = probs < torch.topk(probs, top_k, dim=1)[0][..., -1, None]
        probs[indices_to_remove] = filter_value
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_probs[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    probs = torch.gather(sorted_probs, 1, sorted_indices.argsort(-1))
    
    pred_token = torch.multinomial(probs, 1) # [BATCH_SIZE, 1]
    return pred_token


def top_k_sampling(probs, k=5):
    """
    Sample uniformly from the top-k tokens
    """
    top_k = torch.topk(probs, k, dim=-1)
    top_k_probs, top_k_indices = top_k.values, top_k.indices
    top_k_probs = 1 / top_k_indices.size(-1) * torch.ones_like(top_k_probs)
    top_k_indices = top_k_indices.squeeze()
    sampled_token = torch.multinomial(top_k_probs, 1)
    return top_k_indices[sampled_token]


### Layers and Blocks
class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    PyTorch doesn't support simply bias=False
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    One operation of multi-head self attention (MHSA).
    Calculate Query, Key, Value and pass through MHSA
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # attention heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, get_attn_map=False):
        """
        Compute self attention output to be added to residual stream
        """
        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # efficient attention using Flash Attention CUDA kernels
        if get_attn_map:
            attn_map = self.get_attention(query=q, key=k)
            torch.cuda.empty_cache()

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0,
            is_causal=True)

        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

        if get_attn_map:
            return y, attn_map
        else:
            return y

    def get_attention(self, query, key) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    One self-attention block
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.config = config
        if config.mlp:
            self.mlp = MLP(config)

    def forward(self, x):
        """
        Add to residual stream after self-attention and MLP.
        """
        x = x + self.attn(self.ln_1(x))
        if self.config.mlp:
            x = x + self.mlp(self.ln_2(x))
        return x

    def fwd_to_attn_map(self, x):
        """
        Add to residual stream after self-attention and MLP.
        """
        y, attn_map = self.attn(self.ln_1(x), get_attn_map=True)
        x = x + y
        if self.config.mlp:
            x = x + self.mlp(self.ln_2(x))
        return x, attn_map


### Model definition
class GPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.LM_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.LM_head.weight # Weight tying

        self.apply(self._init_weights) # init
        for pn, p in self.named_parameters(): # apply scaled init to residuals
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    ## Init weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    ## Logit computation
    def forward(self, inputs):
        device = inputs.device
        b, t = inputs.size()

        # Compute position/token embeddings
        tok_emb = self.transformer.wte(inputs)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.LM_head(x)
        return logits 


    ## Eval relevant functions
    # Next token probs
    @torch.no_grad()
    def next_token_probs(self, inputs, logprobs=False, prune_vocab=None):
        """
        Compute the probability of the next token given a batch of inputs
        """
        logits = self.forward(inputs)
        logits = logits[:, -1, :] # [B, L, V] -> [B, 1, V]
        if prune_vocab is not None:
            logits = logits[:, :prune_vocab]
        if logprobs:
            return F.log_softmax(logits, -1)
        else:
            return F.softmax(logits, -1)


    # Per token probs
    @torch.no_grad()
    def per_token_dist(self, inputs, logprobs=False, prune_vocab=None):
        """
        Compute the next-token distribution given a batch of inputs
        """
        logits = self.forward(inputs) # [B, L, V]
        if prune_vocab is not None:
            logits = logits[:, :, :prune_vocab]
        if logprobs:
            return F.log_softmax(logits, -1)
        else:
            return F.softmax(logits, -1)


    # Negative log likelihood of a sequence
    @torch.no_grad()
    def get_loglikelihoods(self, sequences, pad_token_id, reduction_type='sum'):
        """
        Compute likelihood of a batch of input samples and corresponding labels
        """
        inputs, labels = sequences[:, :-1].clone(), sequences[:, 1:].clone()
        labels[labels == pad_token_id] = -100  # Mask padding

        # Compute logits
        logits = self.forward(inputs)

        # Compute NLL
        ll = - F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100, 
            reduction=reduction_type,
            )
        return ll
    
    
    # Sample from the model
    @torch.no_grad()
    def sample(self, inputs, max_new_tokens, n_states=None, 
               sampling_strategy='stochastic', retrieve_llhoods=None):
        """
        Sample from the model given a batch of inputs
        """
        if retrieve_llhoods is not None:
            per_token_llhoods = []

        for t in range(max_new_tokens):
            # Compute probabilities of next token given the sequence 
            probs = self.next_token_probs(inputs, logprobs=False, prune_vocab=n_states)

            # Sample next token
            if sampling_strategy == 'stochastic':
                next_token = torch.multinomial(probs, 1)
            elif sampling_strategy == 'greedy':
                next_token = torch.argmax(probs, -1, keepdims=True)
            elif sampling_strategy == 'top_k_top_p':
                next_token = top_k_top_p_filtering(probs, top_k=5, top_p=0.9)
            elif sampling_strategy == 'top_k':
                next_token = top_k_sampling(probs, k=5)
            else:
                raise ValueError("Invalid sampling strategy")
            
            # Update inputs
            inputs = torch.cat((inputs, next_token), dim=1)

            # Compute log likelihood of the sampled token
            if retrieve_llhoods is not None:
                llhood = torch.log(probs.gather(1, next_token)).squeeze()
                per_token_llhoods.append(llhood)

        # Return the sequence and log likelihoods    
        if retrieve_llhoods == 'tokens':
            per_token_llhoods = torch.stack(per_token_llhoods, dim=1)
            return inputs, per_token_llhoods
        elif retrieve_llhoods == 'sequences':
            llhoods = per_token_llhoods.sum(dim=1)
            return inputs, llhoods
        else:
            return inputs
    

    # Get attention map for a given batch of samples
    @torch.no_grad()
    def get_attention_map(self, inputs):
        """
        Compute attention map for a batch of input samples
        """
        attn_maps = {}
        device = inputs.device
        b, t = inputs.size()

        # Compute position/token embeddings
        tok_emb = self.transformer.wte(inputs)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        for b_id, block in enumerate(self.transformer.h):
            x, attn_map = block.fwd_to_attn_map(x)
            attn_maps[b_id] = attn_map

        return attn_maps