import torch
import numpy as np
import torch.nn.functional as F

def grammar_evals(cfg, model, template, grammar, constraints_results, device):
    """
    Evaluate the model on grammaticality and type constraints.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): Model to evaluate.
        template (torch.Tensor): Template to generate samples from.
        grammar (Grammar): Grammar object.
        constraints_results (dict): Dictionary to store results of type constraints.
        device (torch.device): Device to run on.

    Returns:
        results_dict (dict): Results of the grammaticality evaluation.
        constraints_results (dict): Results of the type constraints.
    """
    model.eval()
    eval_bsize = 128

    with torch.no_grad():

        # Generate samples
        inputs = template.repeat(eval_bsize, 1).to(device)
        samples, per_token_logprobs = model.sample(
            inputs=inputs, 
            max_new_tokens=cfg.data.max_sample_length - 10, 
            retrieve_llhoods='tokens',
            )

        # Transfer to CPU and detokenize
        samples = samples.cpu().numpy()
        samples = [grammar.detokenize_sentence(s).split('<eos>')[0] for s in samples]

        # Eval grammatical correctness
        results_grammaticality = {
            'validity': {'num': 0, 'satisfied': 0},
            'logprobs': {'max': 0, 'min': 0, 'mean': 0, 'distance': 0},
            'depth': {'max': 0, 'min': 0, 'mean': 0},
            'failures': None,
            'model_logprobs': {'max': 0, 'min': 0, 'mean': 0},
            'lengths': {'max': 0, 'min': 0, 'mean': 0}
            }

        logprobs, model_logprobs, depth, lengths = [], [], [], []
        failures = []
        for sid, s in enumerate(samples):
            results_grammaticality['validity']['num'] += 1
            grammaticality, n_tokens = grammar.check_grammaticality(s)
            failures.append(grammaticality[3])
            model_logprobs.append(per_token_logprobs[sid, :n_tokens].sum().item())
            lengths.append(float(n_tokens))

            if grammaticality[0]:
                results_grammaticality['validity']['satisfied'] += 1
                logprobs.append(grammaticality[1])
                depth.append(grammaticality[2])
            else:
                logprobs.append(-5.)
                depth.append(0.)

        logprobs, depth = torch.tensor(logprobs).float(), torch.tensor(depth).float()
        model_logprobs = torch.tensor(model_logprobs).float()
        lengths = torch.tensor(lengths).float()

        # Update results
        results_grammaticality['validity'] = (
            results_grammaticality['validity']['satisfied'] / 
            results_grammaticality['validity']['num']
            )

        results_grammaticality['logprobs']['max'] = logprobs.max().item()
        results_grammaticality['logprobs']['min'] = logprobs.min().item()
        results_grammaticality['logprobs']['mean'] = logprobs.mean().item()
        results_grammaticality['logprobs']['distance'] = (model_logprobs - logprobs).abs().mean().item()

        results_grammaticality['depth']['max'] = depth.max().item()
        results_grammaticality['depth']['min'] = depth.min().item()
        results_grammaticality['depth']['mean'] = depth.mean().item()

        results_grammaticality['model_logprobs']['max'] = model_logprobs.max().item()
        results_grammaticality['model_logprobs']['min'] = model_logprobs.min().item()
        results_grammaticality['model_logprobs']['mean'] = model_logprobs.mean().item()

        results_grammaticality['lengths']['max'] = lengths.max().item()
        results_grammaticality['lengths']['min'] = lengths.min().item()
        results_grammaticality['lengths']['mean'] = lengths.mean().item()

        results_grammaticality['failures'] = failures


        # Eval whether type constraints are satisfied
        current_results = {
            'rel': {'num': 0, 'satisfied': 0},
            'adv': {'num': 0, 'satisfied': 0},
            'desc': {'num': 0, 'satisfied': 0},
            'adj': {'num': 0, 'satisfied': 0}
            }

        # Check type constraints
        for s in samples:
            stats_constraints = grammar.check_type_constraints(s)

            # Update results
            for types in ['rel', 'adv', 'desc', 'adj']:
                if stats_constraints[types][1] is not None:
                    current_results[types]['satisfied'] += stats_constraints[types][0]
                    current_results[types]['num'] += stats_constraints[types][1]

        # Update results if the relevant type was seen in the batch
        for types in ['rel', 'adv', 'desc', 'adj']:
            if current_results[types]['num'] > 0:
                constraints_results[types]['satisfied'] = current_results[types]['satisfied']
                constraints_results[types]['num'] = current_results[types]['num']
                constraints_results[types]['acc'] = (
                    current_results[types]['satisfied'] / current_results[types]['num']
                    )

        results_constraints = {
            'rel_verb_constraints': constraints_results['rel']['acc'],
            'adv_constraints': constraints_results['adv']['acc'],
            'desc_prop_constraints': constraints_results['desc']['acc'],
            'adj_constraints': constraints_results['adj']['acc'],
            'all_constraints': (constraints_results['rel']['acc'] * constraints_results['adv']['acc']
                * constraints_results['desc']['acc'] * constraints_results['adj']['acc']),
        }

        # Results dict
        results_dict = {
            'grammaticality': results_grammaticality,
            'type constraints': results_constraints,
        }

        return results_dict, constraints_results


def llhood_evals(model, symbolic_sentences, grammar, device, max_sample_length, pad_token_id):
    """
    Evaluate the model on loglikelihoods of altered sentences.

    Args:
        model (torch.nn.Module): Model to evaluate.
        symbolic_sentences (list): List of symbolic sentences.
        grammar (Grammar): Grammar object.
        device (torch.device): Device to run on.
        max_sample_length (int): Maximum sequence length.
        pad_token_id (int): ID of the padding token.

    Returns:
        llhoods (dict): Loglikelihoods of altered sentences.
    """

    model.eval()
    with torch.no_grad():

        llhoods = {} # Loglikelihoods

        # Get loglikelihoods for different alteration methods
        for alteration_method in ['normal', 'uniform', 'adversarial', 'partially_adversarial',
                                  'randomize_sentence', 'randomize_values']:

            # Define altered sentences
            sentences = []
            for s in symbolic_sentences:

                ## Alter the sentence
                if alteration_method == 'randomize_sentence' or alteration_method == 'randomize_values':
                    prior_alter_method = 'normal'
                else:
                    prior_alter_method = alteration_method

                sequence, _, _, _ = grammar.fill_sentence_with_values(
                symb_sentence=s,
                n_max_phrases=4,
                s_logprob=1.,
                prior_alter_method=prior_alter_method,
                )

                if alteration_method == 'randomize_values':
                    sequence = grammar.randomize_sent_values(sequence)

                ## Add instruction decorator
                sequence = f'Task: T0 \n Ops: <null> \n Out: {sequence}'

                ## Tokenize the sequence
                sequence = torch.tensor(grammar.tokenize_sentence(sequence))

                ## Randomize sentence, if needed
                if alteration_method == 'randomize_sentence':
                    permuted_sequence = torch.randperm(sequence[7:-1].size(0))
                    sequence = torch.cat((sequence[:7], sequence[7:-1][permuted_sequence], sequence[-1:]))

                # Prune the sequence to the max sequence length
                if sequence.size(0) > max_sample_length:
                    sequence = sequence[:max_sample_length]

                # Pad the sequence to the max sequence length with <pad>
                else:
                    sequence = torch.cat((sequence, torch.tensor(
                        [pad_token_id] * (max_sample_length - len(sequence)))))

                # Append to list of altered sentences
                sentences.append(sequence)

            # Stack sentences
            sentences = torch.stack(sentences, dim=0).to(device)

            # Get loglikelihoods 
            llhoods[alteration_method] = model.get_loglikelihoods(sentences, pad_token_id).item() / sentences.size(0)

    return llhoods


def unscramble_evals(model, symbolic_sentences, grammar, save_tables, cfg_save_tables, device):
    """
    Evaluate the performance of a model in unscrambling symbolic sentences.

    Args:
        model (torch.nn.Module): The model to evaluate.
        symbolic_sentences (list): A list of symbolic sentences to evaluate.
        grammar (Grammar): An instance of the Grammar class.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation results. The dictionary has the following keys:
            - 'per token acc': A dictionary with the average per-token accuracy for each alteration method.
            - 'exact match': A dictionary with the average exact match accuracy for each alteration method.
    """

    model.eval()
    with torch.no_grad():

        per_token_accs = {'normal': [], 'uniform': [], 'adversarial': []}
        exact_accs = {'normal': [], 'uniform': [], 'adversarial': []}
        intersection_accs = {'normal': [], 'uniform': [], 'adversarial': []}
        grammar_check_results = {'normal': [], 'uniform': [], 'adversarial': []}
        type_check_results = {'normal': [], 'uniform': [], 'adversarial': []}
        length_wise_accs = {'<6': [], '7-9': [], '10-12': [], '13-15': [], '16-18': [], '>=19': []}
        save_tables = (save_tables % 10 == 0) and cfg_save_tables
        if save_tables:
            sentences = {'normal': {}, 'uniform': {}, 'adversarial': {}}

        # Get loglikelihoods for different alteration methods
        for alteration_method in ['normal', 'uniform', 'adversarial']:

            # Define altered sentences
            for n, s in enumerate(symbolic_sentences):

                sequence, _, _, _ = grammar.fill_sentence_with_values(
                symb_sentence=s,
                n_max_phrases=4,
                s_logprob=1.,
                prior_alter_method=alteration_method,
                )

                # Define ops for the prompt using the ground truth sequence
                ops = sequence.split()[:-1]
                np.random.shuffle(ops)
                ops = ' '.join(ops)

                # Add instruction decorator
                prompt = f'Task: T1 \n Ops: {ops} \n Out:'

                # Tokenize the sequence
                prompt = torch.tensor(grammar.tokenize_sentence(prompt)).to(device)
                sequence = torch.tensor(grammar.tokenize_sentence(sequence)).to(device)
                prompt_len, seq_len = prompt.size(0), sequence.size(0)

                # Model generations
                generation = model.sample(
                    inputs=prompt.view(1,-1), 
                    max_new_tokens=seq_len, 
                    retrieve_llhoods=None,
                    )
                generation = generation[0][prompt_len:]

                # Compute per token match and overall sequence match accuracy
                matches = (generation == sequence) * 1.
                if seq_len == 0:
                    print(sequence, generation)
                    per_token_accs[alteration_method].append(0)
                    exact_accs[alteration_method].append(0)
                    intersection_accs[alteration_method].append(0)
                    grammar_check = [[0]]
                    type_check = 0
                else:
                    # Evaluate the generated sentence for grammaticality and type constraints
                    per_token_accs[alteration_method].append(matches.mean().item())
                    exact_accs[alteration_method].append(matches.prod().item())
                    generation = grammar.detokenize_sentence(generation.cpu().numpy()).split('<eos>')[0]
                    sequence = grammar.detokenize_sentence(sequence.cpu().numpy()).split('<eos>')[0]
                    grammar_check = grammar.check_grammaticality(generation)
                    type_check = grammar.check_type_constraints(generation)
                    n_t = []
                    for v in type_check.values():
                        n_t += [v[0] / v[1] if v[1] > 0 else 1.]
                    type_check = np.prod(n_t)
                    n_intersection = len(set(generation.split()).intersection(set(sequence.split())))
                    intersection_accs[alteration_method].append(n_intersection / len(set(sequence.split())))

                    # Update length-wise accuracies
                    if seq_len < 6:
                        length_wise_accs['<6'].append(matches.prod().item())
                    elif seq_len < 10:
                        length_wise_accs['7-9'].append(matches.prod().item())
                    elif seq_len < 13:
                        length_wise_accs['10-12'].append(matches.prod().item())
                    elif seq_len < 16:
                        length_wise_accs['13-15'].append(matches.prod().item())
                    elif seq_len < 19:
                        length_wise_accs['16-18'].append(matches.prod().item())
                    else:
                        length_wise_accs['>=19'].append(matches.prod().item())


                # Update results
                grammar_check_results[alteration_method].append(grammar_check[0][0] * 1.)
                type_check_results[alteration_method].append(type_check)

                # Print the prompt, GT sequence, and the generation
                if save_tables:
                    sentences[alteration_method][n] = {'GT': [], 'generated': [], 'per_token': [], 
                                                       'exact': [], 'intersection': [], 'seq_length': []}
                    sentences[alteration_method][n]['seq_length'].append(seq_len)
                    sentences[alteration_method][n]['per_token'].append(per_token_accs[alteration_method][-1])
                    sentences[alteration_method][n]['exact'].append(exact_accs[alteration_method][-1])
                    sentences[alteration_method][n]['intersection'].append(intersection_accs[alteration_method][-1])
                    sentences[alteration_method][n]['GT'].append(sequence)
                    sentences[alteration_method][n]['generated'].append(generation)

        # Convert results to single element
        results_dict = {
            'per token acc': {k: np.mean(v) for k, v in per_token_accs.items()},
            'exact match': {k: np.mean(v) for k, v in exact_accs.items()},
            'intersection acc': {k: np.mean(v) for k, v in intersection_accs.items()},
            'grammaticity': {k: np.mean(v) for k, v in grammar_check_results.items()},
            'type check': {k: np.mean(v) for k, v in type_check_results.items()},
            'sentences': sentences if save_tables else {},
            'length-wise acc': {k: np.mean(v) for k, v in length_wise_accs.items()},
            }

    return results_dict



def cond_gen_evals(model, symbolic_sentences, grammar, max_sample_length, device, save_tables, cfg_save_tables):

    model.eval()
    with torch.no_grad():

        conditioning_satisfied_results = {'normal': [], 'uniform': [], 'adversarial': []}
        grammar_check_results = {'normal': [], 'uniform': [], 'adversarial': []}
        type_check_results = {'normal': [], 'uniform': [], 'adversarial': []}
        save_tables = (save_tables % 10 == 0) and cfg_save_tables
        save_tables = False
        if save_tables:
            sentences = {'normal': {}, 'uniform': {}, 'adversarial': {}}

        # Get loglikelihoods for different alteration methods
        for alteration_method in ['normal', 'uniform', 'adversarial']:

            # Define altered sentences
            for n, s in enumerate(symbolic_sentences):

                sequence, _, _, conditioning_info = grammar.fill_sentence_with_values(
                symb_sentence=s,
                n_max_phrases=4,
                s_logprob=1.,
                prior_alter_method=alteration_method,
                )

                # Define ops for the prompt using the ground truth sequence
                cond_vars = []

                for phrase_id in conditioning_info.keys():
                    # Subject identifiers
                    subjects = conditioning_info[phrase_id]['subjects_idx']
                    if len(subjects) > 0:
                        n_subjects = 1 + np.random.randint(len(subjects))
                        cond_vars.append(list(np.random.choice(subjects, size=n_subjects, replace=False)))

                    # Object identifiers
                    objects = conditioning_info[phrase_id]['objects_idx']
                    if len(objects) > 0:
                        n_objects = 1 + np.random.randint(len(objects))
                        cond_vars.append(list(np.random.choice(objects, size=n_objects, replace=False)))

                    # Verbs
                    verbs = conditioning_info[phrase_id]['verbs']
                    if len(verbs) > 0:
                        n_verbs = 1 + np.random.randint(len(verbs))
                        cond_vars.append(list(np.random.choice(verbs, size=n_verbs, replace=False)))

                    # Descriptors
                    descriptors = conditioning_info[phrase_id]['descriptors']
                    if len(descriptors) > 0:
                        n_descriptors = 1 + np.random.randint(len(descriptors))
                        cond_vars.append(list(np.random.choice(descriptors, size=n_descriptors, replace=False)))

                # Flatten
                np.random.shuffle(cond_vars)
                cond_vars = [v[0] for v in cond_vars]
                ops = ' '.join(cond_vars)

                # Define instruction decorator
                prompt = f'Task: T2 \n Ops: {ops} \n Out:'

                # Tokenize the prompt
                prompt_tokens = torch.tensor(grammar.tokenize_sentence(prompt)).to(device)
                prompt_len = prompt_tokens.size(0)

                # Model generations
                generation = model.sample(
                    inputs=prompt_tokens.view(1,-1), 
                    max_new_tokens=max_sample_length - prompt_len - 10, 
                    retrieve_llhoods=None,
                    )
                generation = generation[0][prompt_len:].cpu().numpy()
                generation = grammar.detokenize_sentence(generation).split('<eos>')[0]

                # Evaluate the generated sentence for grammaticality and type constraints
                conditioning_satisfied = 1
                for v in cond_vars:
                    conditioning_satisfied *= (v in generation) * 1.
                grammar_check = grammar.check_grammaticality(generation)
                type_check = grammar.check_type_constraints(generation)
                n_t = []
                for v in type_check.values():
                    n_t += [v[0] / v[1] if v[1] > 0 else 1.]
                type_check = np.prod(n_t)

                # Update results
                conditioning_satisfied_results[alteration_method].append(conditioning_satisfied)
                grammar_check_results[alteration_method].append(grammar_check[0][0] * 1.)
                type_check_results[alteration_method].append(type_check)

                # Print the prompt, GT sequence, and the generation
                if save_tables:
                    try:
                        sentences[alteration_method][n] = {'generated': [], 'cond_satisfied': [], 
                                                           'grammaticality': [], 'type_check': [], 
                                                           'operands': []}
                        sentences[alteration_method][n]['grammaticality'].append(grammar_check_results[alteration_method][-1])
                        sentences[alteration_method][n]['type_check'].append(type_check_results[alteration_method][-1])
                        sentences[alteration_method][n]['cond_satisfied'].append(conditioning_satisfied_results[alteration_method][-1])
                        sentences[alteration_method][n]['operands'].append(ops)
                        sentences[alteration_method][n]['generated'].append(generation)
                    except:
                        print(alteration_method, n)
                        print('Error:', ops, generation)
                        print(sentences)

        # Convert results to single element
        results_dict = {
            'cond satisfied': {k: np.mean(v) for k, v in conditioning_satisfied_results.items()},
            'grammaticity': {k: np.mean(v) for k, v in grammar_check_results.items()},
            'type check': {k: np.mean(v) for k, v in type_check_results.items()},
            'sentences': sentences if save_tables else {},
            }

    return results_dict



def eval_reachable_pairs(model, grammar, device, K=2000):
    """
    While we expect several transitions underlying our system, the core theory is focused on 
    the system's knowledge of object-property pairs. This evaluation tests the model's ability to
    infer pairs that are unlikely for it to have seen during training. This metric is different from
    accuracy, which asks a model to produce a property given an object (i.e., sampling is conditional).
    Instead, here we ask the model if it believes a given object--property pair can go together.
    To do so, we will look at its loglikelihoods for the pairs and compare them to the loglikelihoods
    see if they fall in the top K completions. We expect this eval to show a transition as training progresses.
    This transition will consequently affect all other tasks where the model needs to generate 
    object-property pairs, yielding transitions over there as well, albeit with a lag and 
    different scaling exponents.

    Args:
        model (torch.nn.Module): The model to evaluate.
        grammar (Grammar): An instance of the Grammar class.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation results. The dictionary has the following keys:
            - 'reachable': Average number of randomly sampled, correct pairs that model believes are reachable 
                and that it should know are reachable.
    """

    model.eval()

    # Eval variables
    n_sentences = 1000
    preambles = {
        'freegen': 'Task: T0 \n Ops: <null> \n Out: ',
        'unscramble': 'Task: T1 \n Ops: {ops} \n Out: ',
    }

    template = '[P] subjectID is descV <eos>'

    # Results dict
    reachability = {
        'freegen': {'av_prob': 0, 'av_rank': 0, 'av_inv_rank': 0, 'percent_topk': 0, 'NLL': 0}, 
        'unscramble': {'av_prob': 0, 'av_rank': 0, 'av_inv_rank': 0, 'percent_topk': 0, 'NLL': 0}
        }

    # Define eval data
    inputs = {'freegen': [], 'unscramble': []}
    for _ in range(n_sentences):
        while True:
            # Generate a sample
            sample = grammar.fill_sentence_with_values(
                    symb_sentence=template,
                    n_max_phrases=4,
                    s_logprob=1.,
                    prior_alter_method='uniform',
                    )[0]

            # Check if the sample has the right number of tokens (allows batching)
            if len(sample.split(' ')) == 6:
                break

        # Free generation sample
        inputs['freegen'].append(grammar.tokenize_sentence(
            preambles['freegen'] + sample
            ))

        # Unscramble sample
        ops = sample.split()[:-1]
        ops = ' '.join(np.array(ops)[np.random.permutation(len(ops))])
        inputs['unscramble'].append(grammar.tokenize_sentence(
            preambles['unscramble'].format(ops=ops) + sample
            ))

    # Stack inputs
    inputs['freegen'] = torch.tensor(inputs['freegen'])
    inputs['freegen'] = inputs['freegen'].to(device)
    inputs['unscramble'] = torch.tensor(inputs['unscramble'])
    inputs['unscramble'] = inputs['unscramble'].to(device)

    # Eval model
    with torch.no_grad():
        for task in ['freegen', 'unscramble']:

            # Logits (remove EOS token)
            logits = model(inputs[task][:, :-1]) # [B, L, V]

            # NLL for entire sequence
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                inputs[task][:,1:].reshape(-1),
                ignore_index=-100, 
                reduction='mean',
                )
            reachability[task]['NLL'] = nll.item()

            # Next token probs right before descV (recall EOS token was removed for logits)
            desc_probs = F.softmax(logits[:, -2, :], dim=-1) # [B, L, V]

            # GT descV token ids (EOS is not removed from the inputs)
            gt_tokenids = inputs[task][:,-2:-1]
            gt_probs = desc_probs.gather(1, gt_tokenids)
            reachability[task]['av_prob'] = gt_probs.mean().item()

            # Rank of GT tokens
            _, sorted_desc_probs_idx = torch.sort(desc_probs, dim=-1, descending=True)
            gt_rank = (sorted_desc_probs_idx == gt_tokenids).nonzero()[:, 1].float() + 1
            reachability[task]['av_rank'] = gt_rank.mean().item() / K
            reachability[task]['av_inv_rank'] = (K / gt_rank).mean().item()
            reachability[task]['percent_topk'] = (gt_rank < 1.5 * K).float().mean().item()

    del inputs
    return reachability