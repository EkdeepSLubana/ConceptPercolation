import numpy as np


def define_prior(prior_size: int, alpha: float = 7e-2, prior_type: str = 'dirichlet'):
    """
    Generate a sparse prior distribution.

    Args:
        prior_size (int): The size of the prior distribution.
        alpha (float, optional): The concentration parameter for the Dirichlet distribution. Default is 0.1.

    Returns:
        numpy.ndarray: A sparse prior distribution.
    """
    if prior_type == 'dirichlet':
        x = np.random.dirichlet(np.repeat(alpha, prior_size), size=1)

    elif prior_type == 'zipfian':
        x = 1 / (np.arange(1, prior_size+1) ** alpha)

    elif prior_type == 'uniform':
        x = np.ones(prior_size)

    elif prior_type == 'structured_zeros':
        x = np.zeros(prior_size)
        x[:int(prior_size * alpha)] = 1

    else:
        raise ValueError(f"Invalid prior type: {prior_type}")

    x = np.random.permutation(x)
    return (x / x.sum()).squeeze()


def alter_prior(prior, prior_alter_method):
    """
    Alter the given prior distribution based on the specified alteration method.

    Args:
        prior (numpy.ndarray): The prior distribution to be altered.
        prior_alter_method (str): The method to alter the prior distribution. Possible values are:
            - 'permute': Permute the elements of the prior distribution randomly.
            - 'reverse': Reverse the order of the elements in the prior distribution.
            - 'uniform': Set all elements of the prior distribution to be uniformly distributed.
            - 'adversarial': Generate an adversarial prior distribution by subtracting each element from 1.
            - Any other value: Return the original prior distribution without any alteration.

    Returns:
        numpy.ndarray: The altered prior distribution, normalized to sum up to 1.
    """
    if prior_alter_method == 'permute':
        prior = np.random.permutation(prior)
    elif prior_alter_method == 'reverse':
        prior = np.flip(prior)
    elif prior_alter_method == 'uniform':
        prior = np.ones_like(prior) / prior.shape[0]
    elif prior_alter_method == 'partially_adversarial':
        prior = 0.5 * prior + 0.5 * (1 - prior)
    elif prior_alter_method == 'adversarial':
        prior = 1 - prior
    else:
        prior = prior
    prior = prior / prior.sum()
    return prior
