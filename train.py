import hydra 
import torch
import torch.nn.functional as F

from model import GPT
from dgp import get_dataloader
from evals import grammar_evals, llhood_evals, unscramble_evals, cond_gen_evals, eval_reachable_pairs

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import sanity_checks, configure_optimizers, update_cosine_warmup_lr
from utils import save_model, move_to_device, log_train, log_eval

@hydra.main(config_path="./config/", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="concept_percolation")
    set_seed(cfg.seed)
    save_config(cfg)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    # Dataloader
    dataloader = get_dataloader(
        n_relative_properties=cfg.data.n_relative_properties,
        n_descriptive_properties=cfg.data.n_descriptive_properties,
        n_descriptive_values=cfg.data.n_descriptive_values,
        num_of_classes_to_divide_over=cfg.data.num_of_classes_to_divide_over,
        prior_param=cfg.data.prior_param,
        props_prior_type=cfg.data.props_prior_type,
        n_entities=cfg.data.n_entities,
        instr_ratio=cfg.data.instr_ratio,
        max_sample_length=cfg.data.max_sample_length,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )

    # Check if model is compatible with data
    sanity_checks(cfg, dataloader.dataset.max_sample_length)

    # Define model
    model = GPT(cfg.model, dataloader.dataset.PCSG.vocab_size)
    model.to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    print("number of parameters: %.2fM" % (model.get_num_params()/1e6,))

    # Optimizer
    optimizer = configure_optimizers(model, cfg.optimizer)

    # Train
    train(cfg, model, dataloader, optimizer, device)

    # Close wandb and log file
    cleanup(cfg, fp)


def train(cfg, model, dataloader, optimizer, device):
    """
    Training function
    """
    # Set model to train mode
    model.train()

    # Set to True to save the grammar underlying the dataset
    save_grammar = True

    # Data type (bf16 for efficiency)
    dt = torch.bfloat16 if cfg.bf16 else torch.float32

    # Configuration
    total_steps = len(dataloader) * cfg.epochs
    train_loss = {'total': []}
    for k in dataloader.dataset.tasks_dict.keys():
        train_loss[k] = []
    lr, it, save_tables = 0.0, 0, 0
    print("Total training steps: ", total_steps)
    print("Learning rate warmup steps: ", cfg.optimizer.warmup_iters)

    # Eval dict
    props_per_class = 360 + 160 * ((cfg.data.n_descriptive_properties - 100) // 40)
    constraints_results = {
        'rel': {'num': 0, 'satisfied': 0, 'acc': 0},
        'adv': {'num': 0, 'satisfied': 0, 'acc': 0},
        'desc': {'num': 0, 'satisfied': 0, 'acc': 0},
        'adj': {'num': 0, 'satisfied': 0, 'acc': 0}
        }

    # Save initial model
    results_dir = save_model(cfg, model, optimizer, it)

    # Training loop
    for _ in range(cfg.epochs):
        for sequences, symb_sequences, seq_lengths, seq_logprobs, _ in dataloader:
            if it > cfg.optimizer.train_iters: # Training destabilizes after a certain point when the loss is too low, so we break
                save_model(cfg, model, optimizer, it)
                break

            # Split sequences into inputs and labels
            # (B, L) -> (B, L-1), (B, L-1)
            B = sequences.size(0)
            inputs, labels = move_to_device([sequences[:,:-1], sequences[:,1:]], device)
            labels[labels == dataloader.dataset.pad_token_id] = -100  # Mask padding

            samples_per_task = {
                k: inputs[:, 1] == dataloader.dataset.task_token_idx[k]
                for k in dataloader.dataset.task_token_idx
            }

            # Sequence statistics to log as well
            train_lengths = {
                'max': seq_lengths.max().item(), 
                'min': seq_lengths.min().item(), 
                'mean': seq_lengths.mean().item()
                }

            # Log train metrics
            if it % cfg.log.log_interval == 0:
                train_loss = log_train(it, cfg.deploy, lr, train_loss, train_lengths)

            # Evals
            if it % cfg.log.eval_interval == 0:
                model.eval() # Set to eval mode

                # Grammaticality results
                grammar_results_dict, constraints_results = grammar_evals(
                    cfg=cfg, model=model, template=dataloader.dataset.template, 
                    grammar=dataloader.dataset.PCSG, device=device,
                    constraints_results=constraints_results
                    ) if cfg.eval.grammar else (None, None)
                
                # Likelihood results
                llhood_results_dict = llhood_evals(
                    model=model,
                    symbolic_sentences=symb_sequences,
                    grammar=dataloader.dataset.PCSG,
                    device=device,
                    max_sample_length=dataloader.dataset.max_sample_length,
                    pad_token_id=dataloader.dataset.pad_token_id,
                ) if cfg.eval.llhood else None

                # Unscrambling results
                unscramble_results_dict = unscramble_evals(
                    model=model,
                    symbolic_sentences=symb_sequences,
                    grammar=dataloader.dataset.PCSG,
                    device=device,
                    save_tables=save_tables,
                    cfg_save_tables=cfg.eval.save_tables,
                ) if cfg.eval.unscramble else None

                # Conditional generation results
                cond_results_dict = cond_gen_evals(
                    model=model,
                    symbolic_sentences=symb_sequences,
                    grammar=dataloader.dataset.PCSG,
                    device=device,
                    max_sample_length=dataloader.dataset.max_sample_length,
                    save_tables=save_tables,
                    cfg_save_tables=cfg.eval.save_tables,
                ) if cfg.eval.cond_gen else None

                # Eval reachable pairs
                reachability_results_dict = eval_reachable_pairs(
                    model=model,
                    grammar=dataloader.dataset.PCSG,
                    device=device,
                    K=props_per_class,
                ) if cfg.eval.reachable_pairs else None

                # Log eval metrics
                save_tables = log_eval(deploy=cfg.deploy, it=it, save_tables=save_tables, 
                            cfg_save_tables=cfg.eval.save_tables,
                            grammaticality_results=grammar_results_dict,
                            llhood_results=llhood_results_dict,
                            unscramble_results=unscramble_results_dict,
                            cond_results_dict=cond_results_dict,
                            reachability_results_dict=reachability_results_dict,
                            ) # Log eval metrics

                model.train() # Set back to train mode

            # Update LR
            it, lr = update_cosine_warmup_lr(it, cfg.optimizer, optimizer, total_steps)

            # Compute loss
            optimizer.zero_grad(set_to_none=True) # Set gradients to None
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=dt): # Mixed precision
                logits = model(inputs)  # (B, L-1, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction='none'
                    ) # (B*L-1)
                loss = loss.reshape(B, -1).mean(dim=1) # Sum over sequence length
                
                # Decompose loss according to task tokens
                for k in dataloader.dataset.task_tokens:
                    train_loss[k].append(loss[samples_per_task[k]].mean().item())

                loss = loss.mean() # Average loss
                train_loss['total'].append(loss.item()) # Log loss

            # Update model
            loss.backward() # Compute gradients
            if cfg.optimizer.grad_clip > 0.0: # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step() # Update weights

            # Save model every few iterations
            if it % cfg.log.save_interval == 0:
                save_model(cfg, model, optimizer, it)
            if save_grammar:
                dataloader.dataset.save_grammar(results_dir)
                save_grammar = False # Save only once
        
        # Save after every epoch
        save_model(cfg, model, optimizer, it)


if __name__ == "__main__":
    main()
