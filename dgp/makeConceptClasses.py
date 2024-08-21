import numpy as np
from .baseClasses import BaseConceptClass
from copy import deepcopy
from .utils import define_prior


def getConceptClasses(
            num_of_classes_to_divide_over: int = 4,
            n_relative_properties: int = 96, 
            n_descriptive_properties: int = 360, 
            n_descriptive_values: int = 25,
            n_entities: int = 20,
            seed: int = None,
            prior_param: float = 5e-2,
            props_prior_type: str = 'dirichlet',
            ) -> None:
        
        """
        Get concept classes for the PCSG.

        Args:
            n_relative_properties: int, number of relative properties
            n_descriptive_properties: int, number of descriptive properties
            n_descriptive_values: int, maximum number of descriptive values
            num_of_classes_to_divide_over: int, number of classes to divide over
            seed: int, random seed

        Returns:
            conceptClasses: dict, concept classes for the PCSG
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        # PII vs. non-PII identifiers
        n_pii_values = int(0.2 * n_entities)
        n_nonpii_values = int(0.8 * n_entities)

        # Define a base concept class with the given numbers of properties
        baseClass = BaseConceptClass(
                n_relative_properties=n_relative_properties, 
                n_descriptive_properties=n_descriptive_properties, 
                n_descriptive_values=n_descriptive_values,
                prior_param=prior_param
            )
        
        # Divide the properties up to a certain depth
        partition_results = dividePropertiesIntoConceptClasses(
                n_descriptive_properties=n_descriptive_properties,
                n_relative_properties=n_relative_properties,
                num_of_classes_to_divide_over=num_of_classes_to_divide_over,
                )
        partitions_map, classIDs_to_properties_map = partition_results
        
        # Identify which classes belong to a property and add this map to the baseClass
        properties_to_classes_map = {}
        for class_id, properties_idx in classIDs_to_properties_map.items():
            for prop_idx in properties_idx:
                prop_name = baseClass.property_idx_to_name[prop_idx]
                if prop_name not in properties_to_classes_map.keys():
                    properties_to_classes_map[prop_name] = []
                properties_to_classes_map[prop_name].append(class_id)
        max_number_of_classes_per_property = max([len(class_ids) for class_ids in properties_to_classes_map.values()])
        for prop_name, class_ids in properties_to_classes_map.items():
            baseClass.all_properties[prop_name].classes = class_ids
            baseClass.all_properties[prop_name].degree_shared = len(class_ids) / max_number_of_classes_per_property

        ##### Non-PII and PII identifiers #####
        # Turn a unique descriptor property into nonPII
        for c in range(0, len(partitions_map.keys())): #, num_of_classes_to_divide_over):
            for prop_idx in np.flip(partitions_map[c]['unique']):
                prop_name = baseClass.property_idx_to_name[prop_idx]
                if 'descriptive' in prop_name:
                    p = prop_name.split('_')[-1]
                    baseClass.all_properties[prop_name].type = 'nonpii_identifier'
                    n = baseClass.all_properties[prop_name].name
                    baseClass.all_properties[prop_name].possible_values = [f'{n}_val_{i}' for i in range(n_nonpii_values)]
                    baseClass.all_properties[prop_name].name = f'nonpii_D{p}'
                    baseClass.all_properties[prop_name].n_nonpii_values = n_nonpii_values
                    baseClass.property_idx_to_name[prop_idx] = f'nonpii_D{p}'
                    break

        # Turn a unique descriptor property into PII for first half of classes
        up_to = len(partitions_map.keys())//2 if n_relative_properties > 0 else len(partitions_map.keys())
        for c in range(0, up_to):
            for prop_idx in np.flip(partitions_map[c]['unique']):
                prop_name = baseClass.property_idx_to_name[prop_idx]
                if 'descriptive' in prop_name:
                    p = prop_name.split('_')[-1]
                    baseClass.all_properties[prop_name].type = 'pii_identifier'
                    baseClass.all_properties[prop_name].name = f'pii_D{p}'
                    baseClass.property_idx_to_name[prop_idx] = f'pii_D{p}'
                    baseClass.all_properties[prop_name].n_pii_values = n_pii_values
                    break

        ##### Recollect properties #####
        baseClass.update_properties_dict()
        for prop in baseClass.all_properties.values():
            prop.redefine_parts_of_speech()
        class_sim_map = get_related_classes(classIDs_to_properties_map, baseClass.num_properties)


        ##### Define concept classes #####
        # Copy the baseClass to define a concept class for each class_id
        # We will change the class's name, update which properties are valid, gather them into relevant lists,
        # update list of parents, and redefine the property values
        conceptClasses = {
            'all': {}, 
            'base': None,
            'subject_classes': [],
            'object_classes': [],
            'classIDs_to_properties_map': classIDs_to_properties_map,
            'descriptive_properties': [],
            'relative_properties': [],
            }
        for class_id, properties_idx in classIDs_to_properties_map.items():
            conceptClasses['all'][class_id] = deepcopy(baseClass)
            conceptClasses['all'][class_id].name = f'ConceptClass_{class_id}'
            conceptClasses['all'][class_id].sim_to_other_classes = class_sim_map[class_id]
            for prop_idx in properties_idx:
                prop_name = baseClass.property_idx_to_name[prop_idx]
                if 'desc' in prop_name and not 'pii' in prop_name:
                    if prop_name not in conceptClasses['descriptive_properties']:
                        conceptClasses['descriptive_properties'].append(prop_name)
                elif 'rel' in prop_name:
                    if prop_name not in conceptClasses['relative_properties']:
                        conceptClasses['relative_properties'].append(prop_name)
                else:
                    pass

                # Turn on properties assigned to the concept class
                conceptClasses['all'][class_id].all_properties[prop_name].is_valid = True
                conceptClasses['all'][class_id].valid_properties.append(prop_name)

                # Update list of identifier properties in the concept class
                if conceptClasses['all'][class_id].all_properties[prop_name].type == 'nonpii_identifier':
                    conceptClasses['all'][class_id].nonpii_identifier_properties.append(prop_name)

                # Update list of PII properties in the concept class
                if conceptClasses['all'][class_id].all_properties[prop_name].type == 'pii_identifier':
                    conceptClasses['all'][class_id].pii_identifier_properties.append(prop_name)

        ### Redefine property values 
        for class_id in conceptClasses['all'].keys():
            for prop_name in conceptClasses['all'][class_id].all_properties.keys():
                if conceptClasses['all'][class_id].all_properties[prop_name].is_valid:
                    # Update the property's value (especially necessary to do for PII)
                    # These values will be updated again later as well to define objects / entities
                    # However, the prior over values for relative properties will be defined here, since 
                    # directionality matters for these properties.
                    if 'rel' in prop_name:
                        if len(conceptClasses['all'][class_id].pii_identifier_properties) > 0:
                            conceptClasses['all'][class_id].all_properties[prop_name].rel_prop_value = 'left'
                            conceptClasses['all'][class_id].all_properties[prop_name].prior_over_values = np.array([1., 0.])
                            baseClass.all_properties[prop_name].class_decompositions['left'].append(class_id)
                            conceptClasses['subject_classes'].append(conceptClasses['all'][class_id].name)
                        else:
                            conceptClasses['all'][class_id].all_properties[prop_name].rel_prop_value = 'right'
                            conceptClasses['all'][class_id].all_properties[prop_name].prior_over_values = np.array([0., 1.])
                            baseClass.all_properties[prop_name].class_decompositions['right'].append(class_id)
                            conceptClasses['object_classes'].append(conceptClasses['all'][class_id].name)
                    else:
                        if 'pii' in prop_name:
                            conceptClasses['all'][class_id].all_properties[prop_name].redefine_possible_identifier_values()

                            # Prior over values for nonPII identifiers
                            if 'non' in prop_name: 
                                conceptClasses['all'][class_id].all_properties[prop_name].prior_over_values = define_prior(
                                    len(conceptClasses['all'][class_id].all_properties[prop_name].possible_values),
                                    alpha=prior_param,
                                    prior_type='uniform'
                                )

                            # Prior over values for PII identifiers
                            else:
                                conceptClasses['all'][class_id].all_properties[prop_name].prior_over_values = define_prior(
                                    n_pii_values,
                                    alpha=prior_param,
                                    prior_type=props_prior_type
                                )

                        # Prior over values for descriptive properties
                        elif 'desc' in prop_name:
                            conceptClasses['all'][class_id].all_properties[prop_name].prior_over_values = define_prior(
                                len(conceptClasses['all'][class_id].all_properties[prop_name].possible_values),
                                alpha=prior_param,
                                prior_type=props_prior_type
                            )
                    
                    # Redefine parts of speech
                    conceptClasses['all'][class_id].all_properties[prop_name].redefine_parts_of_speech()
                    conceptClasses['all'][class_id].all_properties[prop_name].assign_rotation_keys()

            # Segregate concept class's properties into relevant lists
            conceptClasses['all'][class_id].segregate_properties()
            conceptClasses['all'][class_id].define_prior_over_properties()

        # Define priors over classes for relative properties
        for prop_name in baseClass.all_properties.keys():
            if 'rel' in prop_name:
                # Defined over the base class because that's the class we use to sample subjects / objects when filling in a sentence.
                # NOTE: This prior is an artifact of an earlier version of the code and doesn't play a critical role
                # anymore. We thus make it be uniform and keep it here to avoid breaking the code, but it could be removed. 
                baseClass.all_properties[prop_name].prior_over_classes['left'] = define_prior(
                    len(baseClass.all_properties[prop_name].class_decompositions['left']),
                    alpha=0,
                    prior_type='uniform',
                    )
                baseClass.all_properties[prop_name].prior_over_classes['right'] = define_prior(
                    len(baseClass.all_properties[prop_name].class_decompositions['right']),
                    alpha=0,
                    prior_type='uniform',
                    )

        # Add base class to concept classes
        conceptClasses['base'] = baseClass
        conceptClasses['descriptive_properties'] = np.sort(conceptClasses['descriptive_properties'])
        conceptClasses['relative_properties'] = np.sort(conceptClasses['relative_properties'])

        return conceptClasses



def get_related_classes(
        classIDs_to_properties_map: dict,
        num_properties: int,
):
    """
    Compute similarity between classes and properties

    Args:
        classIDs_to_properties_map: A dictionary with class IDs as keys
            and a list of properties as values.
        num_properties: The total number of properties

    Returns:
        class_sim_map: A matrix with similarity between classes.
    """
    # Compute similarity between classes and properties
    sim_vecs = np.zeros((len(classIDs_to_properties_map.keys()), num_properties))
    for k, v in classIDs_to_properties_map.items():
        for p in v:
            sim_vecs[k, p] = 1

    # Compute similarity between classes
    class_sim_vecs = sim_vecs / np.linalg.norm(sim_vecs, axis=1)[:, None]
    class_sim_map = class_sim_vecs @ class_sim_vecs.T - 0.5 * np.eye(class_sim_vecs.shape[0])
    class_sim_map = np.exp(class_sim_map / 0.15)
    class_sim_map = class_sim_map / class_sim_map.sum(axis=1)[:, None]

    return class_sim_map


def gatherConceptClassesVocab(
        conceptClasses: dict,
        ) -> dict:
    """
    Gather all tokens from all properties in all concept classes

    Args:
        conceptClasses: A dictionary with concept class IDs as keys
            and concept classes as values.

    Returns:
        vocab: A dictionary with tokens as keys and indices as values.
    """

    # Gather all tokens from all properties in all concept classes
    vocab = []
    for _, conceptClass in conceptClasses['all'].items():
        if conceptClass.name not in vocab:
            vocab.append(conceptClass.name)
        for _, prop in conceptClass.all_properties.items():
            if prop.is_valid:
                if prop.name not in vocab:
                    vocab.append(prop.name)
                prop_vocab = prop.gather_property_vocab()
                for token in prop_vocab:
                    if token not in vocab:
                        vocab.append(token)

    # Sort the vocab and assign an index to each token
    vocab = {k: v for v, k in enumerate(sorted(vocab))}

    return vocab


def dividePropertiesIntoConceptClasses(
        n_descriptive_properties: int = 50,
        n_relative_properties: int = 50,
        num_of_classes_to_divide_over: int = 3,
        ):
    """
    Partition properties into classes based on depth and the proportion of shared properties.

    Args:
        n_descriptive_properties (int): The number of descriptive properties.
        n_relative_properties (int): The number of relative properties.
        n_shared_desc_properties (int): The number of shared descriptive properties.
        num_of_classes_to_divide_over (int): The number of classes to divide the properties over.

    """

    ## Splitting parameters
    # Shared properties per class
    if n_relative_properties > 0 and num_of_classes_to_divide_over > 1:
        S_rel = int(n_relative_properties / (num_of_classes_to_divide_over // 2)) # Relative ones
    else:
        S_rel = 0

    # Number of unique descriptive properties per class
    U = int(n_descriptive_properties // num_of_classes_to_divide_over)

    # Sanity checks
    assert U * num_of_classes_to_divide_over == n_descriptive_properties, f"{U} * {num_of_classes_to_divide_over} != {n_descriptive_properties}"
    assert S_rel * (num_of_classes_to_divide_over // 2) == n_relative_properties, f"{S_rel} * {num_of_classes_to_divide_over // 2} != {n_relative_properties}"

    ## Split relative properties into groups
    relative_properties = np.arange(n_descriptive_properties, n_descriptive_properties + n_relative_properties)
    if num_of_classes_to_divide_over % 2 == 0:
        relative_properties_groups = np.array_split(relative_properties, num_of_classes_to_divide_over // 2)
        relative_properties_groups = relative_properties_groups + relative_properties_groups

    descriptive_properties = np.arange(n_descriptive_properties)
    # shared_desc_properties = np.random.choice(descriptive_properties, int(n_shared_desc_properties), replace=False)
    # descriptive_properties = np.setdiff1d(descriptive_properties, shared_desc_properties)

    #### Define the partition
    partition = {}
    classes_to_properties_map = {}
    for c in range(num_of_classes_to_divide_over):
        if num_of_classes_to_divide_over % 2 == 0:
            partition[c] = {
                    'shared': np.array(relative_properties_groups[c], dtype=np.int64),
                    'unique': np.random.choice(descriptive_properties, int(U), replace=False),
                    }
        else:
            partition[c] = {
                    'shared': np.array([], dtype=np.int64),
                    'unique': np.random.choice(descriptive_properties, int(U), replace=False),
                    }
        classes_to_properties_map[c] = np.sort(
            np.concatenate((partition[c]['shared'], partition[c]['unique']), axis=0)
            )
        descriptive_properties = np.setdiff1d(descriptive_properties, partition[c]['unique'])

    return partition, classes_to_properties_map