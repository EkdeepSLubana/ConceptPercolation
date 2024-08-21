import numpy as np

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

    Returns:
        Tuple[dict, dict]: A tuple containing the partition and the classes to properties map.
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