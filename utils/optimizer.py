import math
import inspect
import torch
from typing import List


### Move data to device
def move_to_device(vars_to_move: List[torch.Tensor], device):
    """
    Move data to device
    """    
    if len(vars_to_move) == 1:
        return vars_to_move[0].to(device)
    else:
        return [v.to(device) for v in vars_to_move]
        
        
def configure_optimizers(net, optim_cfg):
    """
    Configure the optimizer and the learning rate scheduler
    """
    # Filter out parameters that do not require grad
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # I.e., all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optim_cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=optim_cfg.learning_rate,
        betas=(optim_cfg.beta1, optim_cfg.beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def update_cosine_warmup_lr(it, cfg, optimizer, total_steps):
    """
    Update learning rate with cosine warmup
    """
    it += 1
    lr = cfg.learning_rate

    if cfg.decay_lr:
        if it < cfg.warmup_iters:
            lr = lr * (it) / cfg.warmup_iters
        else:
            num = (it - cfg.warmup_iters)
            decay_ratio = num / (total_steps - cfg.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = cfg.min_lr + coeff * (lr - cfg.min_lr)
        
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return it, lr