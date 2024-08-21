import torch
import wandb
import numpy as np
import random
import os
import sys
import warnings
import yaml
from omegaconf import OmegaConf


# Sanity checks
def sanity_checks(cfg, max_sample_length):
    """
    Basic sanity checks for model configuration and data compatibility
    """

    # Check if vocabulary size and sequence length are compatible
    assert(cfg.model.context_size >= max_sample_length)
    assert(cfg.model.n_embd % cfg.model.n_head == 0)

    # Check if BF16 is supported
    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning)

        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")


# Seed
def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    # rng = np.random.default_rng(seed)
    # true_seed = int(rng.integers(2**30))

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Wandb and logging
def open_log(cfg):
    """
    Open log file and redirect stdout and stderr to it
    """
    print(cfg)
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    if cfg.deploy:
        fname = 'logs/' + cfg.tag + '/' + wandb.run.id + ".log"
        fout = open(fname, "a", 1)
        sys.stdout = fout
        sys.stderr = fout
        print(cfg)
        return fout


def save_config(cfg):
    """
    Save configuration to file
    """
    results_dir = 'results/' + cfg.tag + "/" + wandb.run.id
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/conf.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg), f)


def init_wandb(cfg, project_name):
    """
    Initialize wandb
    """
    if cfg.deploy:
        wandb.init(
            project=project_name,
            entity='sodl'
            )
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()


def log_train(it, deploy, lr, train_loss, train_lengths):
    """
    Log training information
    """
    if deploy and len(train_loss) > 0:
        wandb.log({
            "train": {k: np.mean(v) for k, v in train_loss.items()},
            "iteration": it,
            "lr": lr
            })

        for k, v in train_lengths.items():
            wandb.log({'train': {f'lengths/{k}': v}})

    print("train -- iter: %d, lr: %.6f, loss: %.4f" % (it, lr, np.mean(train_loss['total'])))
    train_loss = {k: [] for k in train_loss.keys()}
    return train_loss


def log_eval(deploy, it, save_tables, cfg_save_tables, grammaticality_results, 
             llhood_results, unscramble_results, cond_results_dict, reachability_results_dict):
    """
    Log eval information
    """

    if deploy:
        wandb.log({'eval': {'iteration': it}})

        # Grammaticality
        if grammaticality_results is not None:
            for key in grammaticality_results['grammaticality'].keys():
                if key == 'failures':
                    continue

                elif key == 'validity':
                    wandb.log({'grammaticality': {'validity': grammaticality_results['grammaticality']['validity']}})

                else:
                    for k, v in grammaticality_results['grammaticality'][key].items():
                        wandb.log({'grammaticality': {f'{key} ({k})': v}})

            # Type constraints
            for key, value in grammaticality_results['type constraints'].items():
                wandb.log({'type constraints': {key: value}})

        # Likelihood results
        if llhood_results is not None:
            for key, value in llhood_results.items():
                wandb.log({'llhoods': {key: value}})

        # Sentence unscrambling results
        if unscramble_results is not None:
            for metric in unscramble_results.keys():
                if metric == 'sentences':
                    continue
                for key, value in unscramble_results[metric].items():
                    wandb.log({f'Unscramble ({metric})': {key: value}})

            if save_tables % 10 == 0 and cfg_save_tables:
                try:
                    table_normal = wandb.Table(columns=['iteration', 'index', 'length', 'per_token', 
                                                'exact', 'GT', 'Gen'])
                    table_adversarial = wandb.Table(columns=['iteration', 'index', 'length', 'per_token', 
                                                'exact', 'GT', 'Gen'])
                    for sent_idx, results in unscramble_results['sentences']['normal'].items():
                        table_normal.add_data(it, sent_idx, results['seq_length'], results['per_token'], 
                                        results['exact'], results['GT'], results['generated'])
                    for sent_idx, results in unscramble_results['sentences']['adversarial'].items():
                        table_adversarial.add_data(it, sent_idx, results['seq_length'], results['per_token'], 
                                        results['exact'], results['GT'], results['generated'])
                    wandb.log({'Unscramble Generations (normal)': table_normal})
                    wandb.log({'Unscramble Generations (adversarial)': table_adversarial})
                except:
                    pass

        # Conditional generation results
        if cond_results_dict is not None:
            for metric in cond_results_dict.keys():
                if metric == 'sentences':
                    continue
                for key, value in cond_results_dict[metric].items():
                    wandb.log({f'Cond gen ({metric})': {key: value}})

            if save_tables % 10 and cfg_save_tables:
                try:
                    table_normal = wandb.Table(columns=['iteration', 'index', 'cond_satisfied', 
                                                'grammaticality', 'type_check', 'operands', 'Gen'])
                    table_adversarial = wandb.Table(columns=['iteration', 'index', 'cond_satisfied', 
                                                'grammaticality', 'type_check', 'operands', 'Gen'])
                    
                    for sent_idx, results in cond_results_dict['sentences']['normal'].items():
                        table_normal.add_data(it, sent_idx, results['cond_satisfied'], results['grammaticality'],
                                        results['type_check'], results['operands'], results['generated'])

                    for sent_idx, results in cond_results_dict['sentences']['adversarial'].items():
                        table_adversarial.add_data(it, sent_idx, results['cond_satisfied'], results['grammaticality'],
                                        results['type_check'], results['operands'], results['generated'])
                        
                    wandb.log({'Cond Generations (normal)': table_normal})
                    wandb.log({'Cond Generations (adversarial)': table_adversarial})
                except:
                    print("Error in saving Cond gen tables")
                    print(cond_results_dict['sentences'])

        # Reachability results
        if reachability_results_dict is not None:
            for task in reachability_results_dict.keys():
                for key, value in reachability_results_dict[task].items():
                    wandb.log({f'Reachability ({task})': {key: value}})


    print("eval -- iter: %d" % it)

    return save_tables+1


# Save model
def save_model(cfg, net, optimizer, it, save_init=False):
    """
    Save model checkpoint
    """
    if cfg.deploy:
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': it,
            'config': cfg,
        }

        fdir = 'results/' + cfg.tag + "/" + wandb.run.id
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, 'ckpt_' + str(it+1) + '.pt')
        else:
            fname = os.path.join(fdir, 'latest_ckpt.pt')
        torch.save(checkpoint, fname)
        return fdir