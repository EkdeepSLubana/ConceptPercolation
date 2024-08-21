from typing import Iterator, List, Tuple, Union
import random
import numpy as np
import nltk  # type: ignore
from nltk.grammar import ProbabilisticProduction  # type: ignore
from nltk.grammar import Nonterminal  # type: ignore
from .makeConceptClasses import getConceptClasses, gatherConceptClassesVocab

Symbol = Union[str, Nonterminal]


class ProbabilisticGenerator(nltk.grammar.PCFG):
    def generate(self, n: int = 1) -> Iterator[str]:
        """Probabilistically, recursively reduce the start symbol `n` times,
        yielding a valid sentence each time.

        Args:
            n: The number of sentences to generate.

        Yields:
            The next generated sentence.
        """
        for _ in range(n):
            x = self._generate_derivation(self.start())
            yield x

    def _generate_derivation(self, nonterminal: Nonterminal) -> str:
        """Probabilistically, recursively reduce `nonterminal` to generate a
        derivation of `nonterminal`.

        Args:
            nonterminal: The non-terminal nonterminal to reduce.

        Returns:
            The derived sentence.
        """
        sentence: List[str] = []
        symbol: Symbol
        derivation: str

        for symbol in self._reduce_once(nonterminal):
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation = self._generate_derivation(symbol)

            if derivation != "":
                sentence.append(derivation)

        return " ".join(sentence)

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        """Probabilistically choose a production to reduce `nonterminal`, then
        return the right-hand side.

        Args:
            nonterminal: The non-terminal symbol to derive.

        Returns:
            The right-hand side of the chosen production.
        """
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabilisticProduction:
        """Probabilistically choose a production that reduces `nonterminal`.

        Args:
            nonterminal: The non-terminal symbol for which to choose a production.

        Returns:
            The chosen production.
        """
        productions: List[ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]



class PCSG:

    def __init__(
            self,
            n_relative_properties = 96,
            n_descriptive_properties = 360,
            n_descriptive_values = 25,
            num_of_classes_to_divide_over = 24,
            prior_param = 5e-2,
            props_prior_type = 'dirichlet',
            n_entities = 25,
            tasks: dict = None,
            seed: int = 42,
            ):
        """Create a probabilistic context-free grammar.

        Args:
            depth: The depth of the concept classes.
            n_relative_properties: The number of relative properties.
            n_descriptive_properties: The number of descriptive properties.
            n_descriptive_values: The maximum number of descriptive values.
            num_of_classes_to_divide_over: The number of classes to divide over.
            r_limits: The limits of the r values.

        Returns:
            A grammar object.
        """

        # Set the random seed
        random.seed(seed)
        np.random.seed(seed)

        # Grammar
        self.production_rules = None
        self.lexical_symbolic_rules = None
        self.grammar = self.create_grammar()

        # Concept classes object
        self.n_relative_properties = n_relative_properties
        self.n_descriptive_properties = n_descriptive_properties
        self.n_descriptive_values = n_descriptive_values
        self.prior_param = prior_param
        self.props_prior_type = props_prior_type
        self.n_entities = n_entities
        self.num_of_classes_to_divide_over = num_of_classes_to_divide_over

        self.conceptClasses = getConceptClasses(
            n_relative_properties=n_relative_properties,
            n_descriptive_properties=n_descriptive_properties,
            prior_param=prior_param,
            props_prior_type=props_prior_type,
            n_descriptive_values=n_descriptive_values,
            num_of_classes_to_divide_over=num_of_classes_to_divide_over,
            n_entities=n_entities,
            )

        # Tasks
        self.tasks = tasks

        # Extract PII / nonPII values and map them to properties for later evals
        self.pii_values_to_prop_map = {}
        self.nonpii_values_to_prop_map = {}
        for _, conceptclass in self.conceptClasses['all'].items():
            for prop in conceptclass.pii_identifier_properties:
                for v in conceptclass.all_properties[prop].possible_values:
                    self.pii_values_to_prop_map[v] = prop
            for prop in conceptclass.nonpii_identifier_properties:
                for v in conceptclass.all_properties[prop].possible_values:
                    self.nonpii_values_to_prop_map[v] = prop

        # Set the vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = self.gather_vocabulary()

        # Record values for later use 
        # (generally used to identify match b/w value and prop for rand value evals)
        self.relative_values = []
        self.descriptive_values = []
        self.adjective_values = []
        self.adverb_values = []

        for t in self.vocab.keys():
            if ('rel' in t) and ('verb' in t):
                self.relative_values.append(t)
            elif ('desc' in t and 'val' in t) and ('pii' not in t):
                self.descriptive_values.append(t)
            elif 'adj' in t:
                self.adjective_values.append(t)
            elif 'adv' in t:
                self.adverb_values.append(t)

        # Parser
        self.parser = nltk.ViterbiParser(self.grammar)


    def create_grammar(self):
        """Create a probabilistic context-free grammar.

        Returns:
            A grammar object.
        """

        # Production rules
        self.production_rules = """
            S -> Ph NP VP EndOfSeq [1.0] | Ph NP VP SepSeq S [0.0]
            NP -> subjectID [0.8] | NP Conj NP [0.2]
            VP -> descPreP descV [0.4] | relV relPreP relNP [0.4] | VP Conj VP [0.2]
            relNP -> objectID [0.7] | objectID Conj relNP [0.3]
            """

        # Symbols for the lexical rules
        self.lexical_symbolic_rules = """
                Ph -> '[P]' [1.0]
                subjectID -> 'subjectID' [1.0]
                objectID -> 'objectID' [1.0]
                relV -> 'relV' [1.0]
                descV -> 'descV' [1.0]
                descPreP -> 'is' [0.5] | 'has' [0.5]
                relPreP -> 'on' [0.4] | 'in' [0.3] | 'to' [0.3]
                Conj -> 'and' [0.5] | 'or' [0.5]
                SepSeq -> '<sep>' [1.0]
                EndOfSeq -> '<eos>' [1.0]
                """

        return ProbabilisticGenerator.fromstring(self.production_rules + self.lexical_symbolic_rules)


    def gather_vocabulary(self):
        """Gather the vocabulary from the concept classes.

        Returns:
            The vocabulary.
        """

        # Gather concept classes' vocabulary
        vocab = gatherConceptClassesVocab(self.conceptClasses)
        vocab_size = len(vocab.keys())

        # Add special tokens from the grammar
        special_lhs = ['SepSeq', 'EndOfSeq', 'descPreP', 'relPreP', 'Conj']
        for lhs, rules in self.grammar._lhs_index.items():
            if str(lhs) in special_lhs:
                for rule in rules:
                    vocab[str(rule.rhs()[0])] = vocab_size
                    vocab_size += 1

        # Add special tokens to be used for defining sequences in dataloader
        for special_token in ['<pad>', 'Task:', '<null>', 'Ops:', 'Out:', '\n']:
            vocab[special_token] = vocab_size
            vocab_size += 1

        # Add task tokens
        for task_token in self.tasks:
            vocab[task_token] = vocab_size
            vocab_size += 1

        # Create an inverse vocabulary
        id_to_token_map = {v: k for k, v in vocab.items()}

        return vocab, id_to_token_map, vocab_size


    def tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
            sentence: The sentence to tokenize.

        Returns:
            The tokenized sentence.
        """

        # Tokenize the sentence
        tokens = sentence.split(' ')

        # Convert the tokens to indices
        token_indices = []
        for token in tokens:
            if token == '' or token == ' ':
                continue
            else:
                token_indices.append(self.vocab[token])

        return token_indices


    def detokenize_sentence(self, token_indices) -> str:
        """Detokenize a sentence.

        Args:
            token_indices: The token indices to detokenize.

        Returns:
            The detokenized sentence.
        """

        # Convert the indices to tokens
        tokens = [self.id_to_token_map[token] for token in np.array(token_indices)]

        # Detokenize the tokens
        sentence = " ".join(tokens)

        return sentence


    def sentence_generator(
            self, 
            num_of_samples: int,
            n_max_phrases: int = 4,
            ) -> Iterator[str]:
        """
        1. Generate a sentence from the grammar
        2. Fill the sentence with values from the concept classes
        """

        # An iterator that dynamically generates symbolic sentences from the underlying PCFG
        symbolic_sentences = self.grammar.generate(num_of_samples)

        # Fill the sentences with values from the concept classes
        for s in symbolic_sentences:
            s_logprob = 0.
            yield self.fill_sentence_with_values(
                symb_sentence=s, 
                n_max_phrases=n_max_phrases, 
                s_logprob=s_logprob,
                prior_alter_method=None,
                )


    def fill_sentence_with_values(
            self, 
            symb_sentence: str, 
            n_max_phrases: int, 
            s_logprob: float, 
            prior_alter_method: str = None,
        ) -> str:
        """
        Fill the sentence with values from the concept classes

        Args:
            sentence: The sentence to fill with values.
            n_max_phrases: The maximum number of phrases in the sentence.

        Returns:
            The sentence filled with values.
        """

        values_logprob = 0.

        ### Set up the phrase maps
        # Split the sentence into phrases
        phrases = symb_sentence.split('[P] ')[1:n_max_phrases+1]

        # Create a map from phrases to their symbols
        phrase_to_symbols_map = {ph_id: ph.split(' ') for ph_id, ph in enumerate(phrases)}
        if len(phrase_to_symbols_map.keys()) > 1:
            phrase_to_symbols_map[0] = phrase_to_symbols_map[0][:-1]

        ### Map phrases to concept classes
        # Choose a base concept class
        class_idx = random.choice(list(self.conceptClasses['all'].keys()))
        values_logprob += np.log(1/len(self.conceptClasses['all'].keys()))
        base_class = self.conceptClasses['all'][class_idx]

        # Assign a concept class to each phrase
        phrase_classes = {k: None for k in phrase_to_symbols_map.keys()}
        for k in phrase_classes.keys():
            if k == 0:
                phrase_classes[k] = base_class
            else:
                class_idx, pval = base_class.sample_similar_class()
                values_logprob += np.log(pval)
                phrase_classes[k] = self.conceptClasses['all'][class_idx]

        ### Fill the sentence with values from the concept classes
        conditioning_info = {}
        for phrase_id in phrase_to_symbols_map.keys():

            conditioning_info[phrase_id] = {
                'subjects_idx': [], 
                'objects_idx': [], 
                'properties': [], 
                'verbs': [], 
                'descriptors': [], 
                'descriptor_subjs': [],
                'adverbs': [], 
                'adjectives': [],
                }

            ## Phrase details
            phrase_class = phrase_classes[phrase_id]
            phrase = phrase_to_symbols_map[phrase_id]
            working_phrase = np.array(phrase)

            # Define PoS
            where_rel_verbs = np.where(working_phrase == 'relV')[0]
            where_desc_verbs = np.where(working_phrase == 'descV')[0]
            where_subject_ids = np.where(working_phrase == 'subjectID')[0]
            where_object_ids = np.where(working_phrase == 'objectID')[0]

            # Define subject / object class vars (phrase class will play whatever its relevant role here is)
            is_subject_class = phrase_class.name in self.conceptClasses['subject_classes']
            subject_class, object_class = (phrase_class, None) if is_subject_class else (None, phrase_class)

            # If there are no relative verbs, then the subject class is the phrase class
            if where_rel_verbs.shape[0] == 0:
                subject_class = phrase_class

            ## Fill relative VPs (will fill rel_verbs, object_ids)
            object_keys = {} # rotation key, value of object
            if where_rel_verbs.shape[0] > 0:

                # Define a map of relative verbs to their positions in the phrase and objects that follow them
                rel_verb_ids = {vnum: {'verb_loc': v_id} for vnum, v_id in enumerate(where_rel_verbs)}
                if where_rel_verbs.shape[0] > 0:
                    for vnum in range(where_rel_verbs.shape[0]-1):
                        last_verb_loc, next_verb_loc = rel_verb_ids[vnum]['verb_loc'], rel_verb_ids[vnum+1]['verb_loc']
                        rel_verb_ids[vnum]['obj_locs'] = where_object_ids[
                            np.where((where_object_ids >= last_verb_loc) & (where_object_ids < next_verb_loc))[0]
                            ]

                # last verb has to be handled separately as there is no next verb to map to
                rel_verb_ids[where_rel_verbs.shape[0]-1]['obj_locs'] = where_object_ids 

                # Fill values for relative VPs
                for verb_num, verb_deets in rel_verb_ids.items():
                    rel_prop_name, pval = phrase_class.sample_relative_property()
                    values_logprob += np.log(pval)
                    conditioning_info[phrase_id]['properties'].append(rel_prop_name)

                    rel_verb_value, pval = phrase_class.all_properties[rel_prop_name].sample_relverb_phrase()
                    values_logprob += np.log(pval)
                    phrase[verb_deets['verb_loc']] = rel_verb_value

                    if 'adv' in rel_verb_value:
                        vals = rel_verb_value.split()
                        conditioning_info[phrase_id]['adverbs'].append(vals[0])
                        conditioning_info[phrase_id]['verbs'].append(vals[1])
                    else:
                        conditioning_info[phrase_id]['verbs'].append(rel_verb_value)

                    # If phrase_class is a subject class, then we need to choose an object class
                    if is_subject_class:
                        object_class, pval = self.conceptClasses['base'].all_properties[rel_prop_name].sample_object_class()
                        values_logprob += np.log(pval)
                        object_class = self.conceptClasses['all'][object_class]

                    # If phrase_class is not a subject class, then we need to choose a subject class
                    else:
                        if verb_num == 0:
                            subject_class, pval = self.conceptClasses['base'].all_properties[rel_prop_name].sample_subject_class()
                            values_logprob += np.log(pval)
                            subject_class = self.conceptClasses['all'][subject_class]

                    # Fill object IDs (uniform sampling)
                    for obj_id in verb_deets['obj_locs']:
                        obj_value, pval, rotation_key = object_class.sample_identifier_value()
                        values_logprob += np.log(pval)
                        object_keys[obj_id] = rotation_key
                        phrase[obj_id] = obj_value
                        if 'adj' in obj_value:
                            vals = obj_value.split(' ')
                            conditioning_info[phrase_id]['adjectives'].append(vals[0])
                            conditioning_info[phrase_id]['objects_idx'].append(vals[1])
                        else:
                            conditioning_info[phrase_id]['objects_idx'].append(obj_value)

            ## Fill subjects (conditioned on which objects were seen; uniform if no objects seen)
            # Subject class is chosen class if desc VP; else it is sampled while filling rel VPs
            subject_keys = {} # rotation key, value of subject
            for subject_idx in where_subject_ids:
                if len(object_keys) > 0:
                    obj_loc = where_object_ids[np.where(where_object_ids > subject_idx)[0][0]]
                    rotation_key = object_keys[obj_loc]
                    subject_value, pval, rotation_key = subject_class.sample_identifier_value(
                        rotation_key=rotation_key
                        )
                else:
                    rotation_key = 0
                    subject_value, pval, rotation_key = subject_class.sample_identifier_value(
                        rotation_key=rotation_key
                        )
                values_logprob += np.log(pval)
                phrase[subject_idx] = subject_value
                if 'adj' in subject_value:
                    vals = subject_value.split()
                    conditioning_info[phrase_id]['adjectives'].append(vals[0])
                    conditioning_info[phrase_id]['subjects_idx'].append(vals[1])
                    subject_keys[subject_idx] = (rotation_key, vals[1])
                else:
                    conditioning_info[phrase_id]['subjects_idx'].append(subject_value)
                    subject_keys[subject_idx] = (rotation_key, subject_value)

            ## Fill descriptive properties
            if where_desc_verbs.shape[0] > 0:

                # Fill values for descriptive VPs
                for desc_verb_loc in where_desc_verbs:
                    desc_prop_name, pval = subject_class.sample_descriptive_property()
                    values_logprob += np.log(pval)
                    conditioning_info[phrase_id]['properties'].append(desc_prop_name)

                    subj_loc = where_subject_ids[np.where(where_subject_ids < desc_verb_loc)[0][-1]]
                    rotation_key, rotation_subj = subject_keys[subj_loc]
                    desc_prop_value, pval, _ = subject_class.all_properties[desc_prop_name].sample_descproperty_value(
                        rotation_key=rotation_key,
                        prior_alter_method=prior_alter_method
                        )
                    values_logprob += np.log(pval)
                    phrase[desc_verb_loc] = desc_prop_value
                    conditioning_info[phrase_id]['descriptor_subjs'].append(rotation_subj)
                    if 'adj' in desc_prop_value:
                        vals = desc_prop_value.split()
                        conditioning_info[phrase_id]['adjectives'].append(vals[0])
                        conditioning_info[phrase_id]['descriptors'].append(vals[1])
                    else:
                        conditioning_info[phrase_id]['descriptors'].append(desc_prop_value)

            ## Fill the phrase
            phrases[phrase_id] = " ".join(phrase)

        ## Clean the sentence
        sentence = " ".join(phrases)
        sentence = sentence.replace(' <sep>  ', ' <sep> ')
        if sentence[-6:] == '<sep> ':
            sentence = sentence.rsplit('<sep> ', 1)
            sentence = sentence[0] + '<eos>'

        ## Return the filled sentence
        return sentence, symb_sentence, values_logprob + s_logprob, conditioning_info
    

    def check_grammaticality(self, sentence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Remove decorators
        if 'Out:' in sentence:
            sentence = sentence.split('Out: ')
            sentence = sentence[1] if len(sentence) > 1 else sentence[0]

        # Tokenize the sentence
        tokens = sentence.split(' ')
        if '' in tokens:
            tokens.remove('')

        # Reformat sentence to grammar symbols
        symbols = ['[P]']
        relv_has_occured = False
        for tok in tokens:
            if tok == '' or tok == ' ':
                continue
            elif 'rel' in tok:
                relv_has_occured = True
                symbols += ['relV']
            elif 'pii' in tok:
                if relv_has_occured:
                    symbols += ['objectID']
                else:
                    symbols += ['subjectID']
            elif 'desc' in tok:
                symbols += ['descV']
            elif 'adj' in tok or 'adv' in tok:
                continue
            elif tok == '<sep>':
                relv_has_occured = False
                symbols += ['<sep>']
                symbols += ['[P]']
            elif tok in self.pii_values_to_prop_map.keys():
                symbols += ['subjectID']
            else:
                symbols += [tok]
        symbols += ['<eos>']

        # Parse the symbols
        try:
            parser_output = self.parser.parse(symbols).__next__()
            logprobs, height = parser_output.logprob(), parser_output.height()
            return (True, logprobs, height, None), len(tokens)
        except:
            failure = ' '.join(symbols)
            return (False, None, None, failure), len(tokens)


    def check_type_constraints(self, sentence: str) -> bool:
        """Check if a sentence follows type constraints of the PCSG.

        Args:
            sentence: The sentence to check.

        Returns:
            Possibility, correctness of the sentence.
        """

        if 'Out:' in sentence:
            phrases = sentence.split('Out: ')
            phrases = phrases[1] if len(phrases) > 1 else phrases[0]
            phrases = phrases.split(' <sep> ')
        else:
            phrases = sentence.split(' <sep> ')
        props_per_phrase = {n: {} for n in range(len(phrases))}

        for phrase_id, phrase in enumerate(phrases):
            tokens = phrase.split(' ')

            n_subjects = 0
            subjects, objects, relverbs, descverbs = [], [], [], []
            relv_has_occured = False

            for tok_idx, tok in enumerate(tokens):
                
                # Relative verbs
                if 'rel' in tok:
                    relv_has_occured = True
                    prop_name = tok.split('_verb')[0]
                    cl = self.conceptClasses['base'].all_properties[prop_name].class_decompositions['left']
                    cr = self.conceptClasses['base'].all_properties[prop_name].class_decompositions['right']
                    valid_advs = self.conceptClasses['base'].all_properties[prop_name].verb['modifiers']

                    t = {
                        'prop_type': 'rel',
                        'prop_name': tok.split('_verb')[0], 
                        'adv': '',
                        'adv_constraint': 0.,
                        'subj_classes': cl,
                        'obj_classes': cr,
                        }
                    
                    if tok_idx > 0:
                        if 'adv' in tokens[tok_idx-1]:
                            t['adv'] = tokens[tok_idx-1]
                        if t['adv'] == '':
                            t['adv_constraint'] = None
                        elif t['adv'] in valid_advs:
                            t['adv_constraint'] = 1
                    relverbs.append(t)


                # NonPII Identifiers
                elif 'nonpii' in tok and 'desc' in tok:
                    prop_name = self.nonpii_values_to_prop_map[tok]
                    classes = self.conceptClasses['base'].all_properties[prop_name].classes
                    valid_adjs = self.conceptClasses['base'].all_properties[prop_name].modifiers['values']
                    t = {
                        'prop_type': 'nonpii',
                        'prop_name': prop_name, 
                        'adj': '',
                        'adj_constraint': 0.,
                        'is_subject': not relv_has_occured,
                        'classes': classes,
                        }

                    if tok_idx > 0:
                        if 'adj' in tokens[tok_idx-1]:
                            t['adj'] = tokens[tok_idx-1]
                        if t['adj'] == '':
                            t['adj_constraint'] = None
                        elif t['adj'] in valid_adjs:
                            t['adj_constraint'] = 1

                    if t['is_subject']:
                        subjects.append(t)
                        n_subjects += 1
                    else:
                        objects.append(t)


                # Descriptive verbs
                elif 'desc' in tok:
                    prop_name = tok.split('_val_')[0]
                    classes = self.conceptClasses['base'].all_properties[prop_name].classes
                    if n_subjects > 0:
                        constraint_satisfied = np.intersect1d(classes, subjects[n_subjects-1]['classes'])
                    else:
                        constraint_satisfied = np.array([])
                    valid_adjs = self.conceptClasses['base'].all_properties[prop_name].modifiers['values']
                    t = {
                        'prop_type': 'descriptive',
                        'prop_name': prop_name, 
                        'adj': '',
                        'adj_constraint': 0.,
                        'classes': classes,
                        'subject_id': n_subjects,
                        'constraint_not_satisfied': len(constraint_satisfied) == 0,
                        }

                    if tok_idx > 0:
                        if 'adj' in tokens[tok_idx-1]:
                            t['adj'] = tokens[tok_idx-1]
                        if t['adj'] == '':
                            t['adj_constraint'] = None
                        elif t['adj'] in valid_adjs:
                            t['adj_constraint'] = 1

                    descverbs.append(t)

                # PII Identifiers
                elif tok in self.pii_values_to_prop_map.keys():
                    prop_name = self.pii_values_to_prop_map[tok]
                    classes = self.conceptClasses['base'].all_properties[prop_name].classes
                    t = {
                        'prop_type': 'pii',
                        'prop_name': prop_name,
                        'adj': '',
                        'classes': classes,
                        }
                    subjects.append(t)
                    n_subjects += 1

                # Phrase Separators
                elif tok == '<sep>':
                    relv_has_occured = False

            props_per_phrase[phrase_id] = {
                'subjects': subjects,
                'objects': objects,
                'relverbs': relverbs,
                'descverbs': descverbs,
            }


        # Check constraints
        n_rel_constraints, satisfied_rel_constraints = 0, 0.
        n_adv_constraints, satisfied_adv_constraints = 0, 0.
        n_desc_constraints, satisfied_desc_constraints = 0, 0.
        n_adj_constraints, satisfied_adj_constraints = 0, 0.

        for phrase_id, props in props_per_phrase.items():

            if len(props['relverbs']) > 0:
                constraint_not_satisfied = False
                for verb in props['relverbs']:
                    n_rel_constraints += 1
                    for subj in props['subjects']:
                        
                        c_subj = np.intersect1d(verb['subj_classes'], subj['classes'])
                        if len(c_subj) == 0:
                            constraint_not_satisfied = True
                            break

                    for obj in props['objects']:
                        c_obj = np.intersect1d(verb['obj_classes'], obj['classes'])
                        if len(c_obj) == 0:
                            constraint_not_satisfied = True
                            break

                    if not constraint_not_satisfied:
                        satisfied_rel_constraints += 1

                    if verb['adv_constraint'] is not None:
                        n_adv_constraints += 1
                        satisfied_adv_constraints += verb['adv_constraint']

            if len(props['descverbs']) > 0:
                for verb in props['descverbs']:
                    n_desc_constraints += 1
                    if not verb['constraint_not_satisfied']:
                        satisfied_desc_constraints += 1

                    if verb['adj_constraint'] is not None:
                        n_adj_constraints += 1
                        satisfied_adj_constraints += verb['adj_constraint']

        stats_constraints = {}
        stats_constraints['rel'] = (satisfied_rel_constraints, n_rel_constraints)
        stats_constraints['adv'] = (satisfied_adv_constraints, n_adv_constraints)
        stats_constraints['desc'] = (satisfied_desc_constraints, n_desc_constraints)
        stats_constraints['adj'] = (satisfied_adj_constraints, n_adj_constraints)

        return stats_constraints


    def randomize_sent_values(self, sequence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Tokenize the sentence
        tokens = sequence.split(' ')
        if '' in tokens:
            tokens.remove('')

        # Reformat sentence to grammar symbols
        new_tokens = []
        for tok in tokens:
            if tok == '' or tok == ' ':
                continue
            elif 'rel' in tok:
                new_tokens += [np.random.choice(self.relative_values)]
            elif tok in self.nonpii_values_to_prop_map.keys():
                new_tokens += [np.random.choice(list(self.nonpii_values_to_prop_map.keys()))]
            elif 'desc' in tok:
                new_tokens += [np.random.choice(self.descriptive_values)]
            elif 'adj' in tok:
                new_tokens += [np.random.choice(self.adjective_values)]
            elif 'adv' in tok:
                new_tokens += [np.random.choice(self.adverb_values)]
            elif tok in self.pii_values_to_prop_map.keys():
                new_tokens += [np.random.choice(list(self.pii_values_to_prop_map.keys()))]
            elif tok == '<sep>' or tok == '<eos>':
                new_tokens += [tok]
            else:
                new_tokens += [tok]

        return ' '.join(new_tokens)