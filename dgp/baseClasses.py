from typing import List, Dict, Any
import numpy as np
import random
import string
from typing import List
from .utils import alter_prior, define_prior


class BasePropertyClass:
    def __init__(
            self, 
            name: str, 
            range_of_values: int = 0, 
            is_valid: bool = False,
            is_pii_identifier: bool = False, 
            is_nonpii_identifier: bool = False,
            seed: int = None,
            prior_param: float = 5e-2
            ) -> None:
        """
        Args:
            name: The name of the property.
            range_of_values: The number of possible values the property can take.        
            is_valid: Whether the property is valid or not.
            is_pii_identifier: Whether the property is personally identifiable information (e.g., name).
            is_nonpii_identifier: Whether the property can be used as an identifier.
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.is_valid = is_valid # Whether the property is valid or not (not defined by default)
        self.name = name # Name of the property
        self.prior_param = prior_param # Dirichlet alpha parameter
        self.classes = [] # Classes which the property belongs to
        self.degree_shared = 0. # Degree of sharedness of the property

        # Property type
        self.type = None # Type of the property
        if 'desc' in name:
            self.possible_values = [f'{name}_val_{i}' for i in range(range_of_values)] # Possible values of the property
            if is_pii_identifier:
                self.type = 'pii_identifier'
                self.possible_values = None # Possible values for the PII identifier
                self.n_pii_values = None # Number of possible values for the PII identifier
            elif is_nonpii_identifier:
                self.type = 'nonpii_identifier'
                self.n_nonpii_values = None # Number of possible values for the non-PII identifier
            else:
                self.type = 'descriptive'
        elif 'rel' in name:
            self.type = 'relative'
            self.rel_prop_value = None # Value of the relative property
            self.possible_values = ['left', 'right'] # Possible values for the relative property
            self.class_decompositions = {'left': [], 'right': []} # Class decompositions
            self.prior_over_classes = {'left': None, 'right': None} # Prior over classes
        else:
            raise ValueError('Invalid property type')

        # Prior over property values (will be redefined later)
        self.prior_over_values = define_prior(
            prior_size=range_of_values, 
            prior_type='uniform')

        # Verb and modifiers
        self.verb, self.modifiers = None, None
        # self.redefine_parts_of_speech()

        # Assign a key for prior rotation to each value of the property
        self.rotation_keys = {}


    def gather_property_vocab(self) -> List[str]:
        """
        Gather the vocabulary of the property.
        """
        vocab = []

        ## Verbs
        if self.type == 'relative':
            vocab += [self.verb['name']]
            if '' in vocab: # Remove empty string
                vocab.remove('')

            vocab += self.verb['modifiers']
            if '' in vocab: # Remove empty string
                vocab.remove('')

        ## Modifiers
        if self.type == 'descriptive' or self.type == 'nonpii_identifier':
            vocab += self.modifiers['values']
            if '' in vocab: # Remove empty string
                vocab.remove('')

        ## Values
        if self.type == 'descriptive' or self.type == 'pii_identifier' or self.type == 'nonpii_identifier':
            vocab += self.possible_values

        return vocab


    def redefine_possible_identifier_values(self) -> None:
        """
        Randomly change the range of values of this property.
        """
        # If the property is PII, we must also redefine the possible values
        if self.type == 'pii_identifier':
            self.possible_values = [
                ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                for _ in range(self.n_pii_values)
                ]
        # If the property is nonPII, we must reformat the possible values
        elif self.type == 'nonpii_identifier':
            self.possible_values = ['nonpii_'+val for val in self.possible_values]
        else:
            raise ValueError('Invalid property type for redefining possible identifier values.')


    def assign_rotation_keys(self) -> None:
        """
        Assign rotation keys to each value of the property.
        """
        for idx, val in enumerate(self.possible_values):
            self.rotation_keys[val] = idx


    def redefine_parts_of_speech(self) -> None:
        """
        Redefine the parts of speech of the property.
        """
        modifier_group = int(self.degree_shared * 4)
        self.verb = self.define_verb(modifier_group=modifier_group)
        self.modifiers = self.define_modifiers(modifier_group=modifier_group)


    def define_verb(self, modifier_group: int = None) -> None:
        """
        Define the verb for the property.

        Args:
            modifier_group: The group of modifiers that the verb belongs to.

        Returns:
            verb: The verb for the property.
        """

        # If the property is an identifier, notion of verb doesn't make sense
        if 'identifier' in self.type or 'desc' in self.type: 
            return {'name': '', 'modifiers': [''], 'prior_over_modifiers': [1.]}

        # Verb dict: initialize with name
        verb = {'name': self.name + '_verb'}

        # Modifiers
        range_of_values = 20
        verb['modifiers'] = [''] + [f'G{modifier_group}_adv{verb_idx}' for verb_idx in range(range_of_values)]
        verb['prior_over_modifiers'] = define_prior(prior_size=range_of_values+1, prior_type='uniform')

        return verb


    def define_modifiers(self, modifier_group: int = None) -> List[str]:
        """
        Define the modifiers for the property.

        Args:
            modifier_group: The group of modifiers that the verb belongs to.

        Returns:
            modifiers: The modifiers for the property.
        """

        # If the property is a PII identifier, modifiers don't make sense
        # If the property is relative, adverbs will be used instead of adjectives
        if self.type == 'pii_identifier' or self.type == 'relative':
            return ['']
        
        # Define range of values: for descriptive properties or nonpii identifiers
        elif self.type == 'descriptive' or self.type == 'nonpii_identifier':
            num_of_modifiers = 20

        # If the property is an identifier, notion of verb doesn't make sense
        else:
            raise ValueError('Invalid property type')

        # Define modifier names 
        modifiers = {}
        modifiers['values'] = [''] + [f'G{modifier_group}_adj{idx}' for idx in range(num_of_modifiers)]

        # Define prior over modifiers for each identifier value
        null_modifier_prior = np.array([0.1])
        group_modifiers_prior = 0.9 * define_prior(
            prior_size=num_of_modifiers, 
            prior_type='uniform')
        modifiers['prior_over_modifiers'] = np.concatenate((null_modifier_prior, group_modifiers_prior))

        return modifiers
    

    def sample_descproperty_value(self, prior_alter_method=None, rotation_key=None, get_key=False) -> None:
        """
        Sample a value for the property.
        """
        # Value
        prior = self.prior_over_values
        prior_rot = np.roll(prior, rotation_key % len(prior)) # Rotate the prior
        prior = alter_prior(prior_rot, prior_alter_method) # Alter the rotated prior
        prop_value = np.random.choice(self.possible_values, p=prior) # Sample value
        pval = prior_rot[self.possible_values.index(prop_value)] # Prior prob of sampled value
        return_key = self.rotation_keys[prop_value] if get_key else None

        # Modifier
        if self.type == 'pii_identifier':
            pval = pval * 1.
        else:
            prior_rot = np.roll(
                self.modifiers['prior_over_modifiers'], 
                self.rotation_keys[prop_value] % len(self.modifiers['prior_over_modifiers'])
                )
            prior = alter_prior(prior_rot, prior_alter_method)
            modifier = np.random.choice(self.modifiers['values'], p=prior)
            pval = pval * prior_rot[self.modifiers['values'].index(modifier)]

        if self.type != 'pii_identifier' and modifier != '':
            prop_value = modifier + ' ' + prop_value

        return prop_value, pval, return_key


    def sample_relverb_phrase(self):
        """
        Sample a verb for the property.
        """
        prior = self.verb['prior_over_modifiers']
        vmod = np.random.choice(self.verb['modifiers'], p=prior)
        pval = self.verb['prior_over_modifiers'][self.verb['modifiers'].index(vmod)]
        return vmod + ' ' + self.verb['name'] if vmod != '' else self.verb['name'], pval


    def sample_object_class(self):
        """
        Sample a object class.
        """
        class_idx = self.class_decompositions['right'][0]
        pval = 1.
        return class_idx, pval


    def sample_subject_class(self):
        """
        Sample a subject class.
        """
        class_idx = self.class_decompositions['left'][0]
        pval = 1.
        return class_idx, pval



class BaseConceptClass:

    def __init__(
            self, 
            name: str = 'general',
            n_relative_properties: int = 50, 
            n_descriptive_properties: int = 50, 
            n_descriptive_values: int = 25,
            prior_param: float = 5e-2
            ) -> None:
        """
        A concept class is a collection of properties and together they define a concept.
        This is the general class that all concept classes will inherit from later. 
        Accordingly, properties will be set to be invalid by default and will be
        assigned to specific classes later by turning the is_valid feature True.

        Args:
            name: The name of the concept class.
            n_relative_properties: Number of properties that are relative 
                    (such properties are binary, with value indicating directionality)
            n_descriptive_properties: The number of properties that are 
                    descriptive (can have multiple values).
        """
        self.name = name 
        self.prior_param = prior_param
        self.n_descriptive_values = n_descriptive_values
        self.sim_to_other_classes = None

        # Descriptive properties
        descriptive_properties = {
            f'descriptive_{i}':
            BasePropertyClass(
                name=f'descriptive_{i}', 
                range_of_values=self.n_descriptive_values,
                is_valid=False,
                is_pii_identifier=False,
                is_nonpii_identifier=False,
                prior_param=prior_param,
                ) for i in range(n_descriptive_properties)
        }
        
        # Relative properties
        relative_properties = {
            f'relative_{i}':
            BasePropertyClass(
                name=f'relative_{i}', 
                range_of_values=2,
                is_valid=False,
                is_pii_identifier=False,
                is_nonpii_identifier=False,
                prior_param=prior_param,
                ) for i in range(n_relative_properties)
        }

        # All properties in the concept class
        # Descriptive properties are listed first, followed by relative properties
        # This helps easily turn a few of the descriptive properties into identifiers later on.
        self.all_properties = {**descriptive_properties, **relative_properties}
        
        # Initialize variables to track properties
        self.valid_properties = []
        self.nonpii_identifier_properties = []
        self.pii_identifier_properties = []
        self.descriptive_properties = []
        self.relative_properties = []
        self.identifiers = []
        self.prior_over_properties = {'descriptive': None, 'relative': None, 'identifiers': None}

        # Property idx to property name mapping
        self.property_idx_to_name = {
            i: prop_name for i, prop_name in enumerate(self.all_properties.keys())
        }

        # Total number of properties in the concept class
        self.num_properties = len(self.all_properties.keys())


    def update_properties_dict(self) -> Dict[str, Any]:
        """
        Collect all properties in the concept class.
        """
        # Update properties dict
        new_properties = {}
        for prop_name, prop in self.all_properties.items():
            new_properties[prop.name] = prop
        self.all_properties = new_properties

        # Update properties idx to name map
        self.property_idx_to_name = {
            i: prop_name for i, prop_name in enumerate(self.all_properties.keys())
        }


    def segregate_properties(self) -> None:
        """
        Recollect all properties in the concept class.
        """
        for prop_name, prop in self.all_properties.items():
            if prop.is_valid:
                if prop.type == 'relative':
                    self.relative_properties.append(prop_name)
                elif prop.type == 'descriptive':
                    self.descriptive_properties.append(prop_name)
                elif prop.type == 'pii_identifier':
                    if prop_name not in self.pii_identifier_properties:
                        self.pii_identifier_properties.append(prop_name)
                elif prop.type == 'nonpii_identifier':
                    if prop_name not in self.nonpii_identifier_properties:
                        self.nonpii_identifier_properties.append(prop_name)
        self.identifiers = self.nonpii_identifier_properties + self.pii_identifier_properties


    def define_prior_over_properties(self) -> np.array:
        """
        Return the prior over properties.
        Prior over properties must be uniform, else some properties may not be seen at all (or very minimally)
        The point of our work is that percolation occurs over object--properties' interactions, not properties themselves
        """

        # Prior over descriptive properties
        self.prior_over_properties['descriptive'] = define_prior(
            prior_size=len(self.descriptive_properties), 
            prior_type='uniform',
        )

        # Prior over relative properties
        self.prior_over_properties['relative'] = define_prior(
            prior_size=len(self.relative_properties), 
            prior_type='uniform',
        )

        # Prior over identifier properties (it's just uniform :D)
        nz = []
        for prop in self.identifiers:
            nz += [len(self.all_properties[prop].possible_values)]
        self.prior_over_properties['identifiers'] = np.array(nz)
        self.prior_over_properties['identifiers'] = self.prior_over_properties['identifiers'] / np.sum(self.prior_over_properties['identifiers'])


    def sample_similar_class(self, sample_most_similar=None) -> int:
        """
        Sample a related class.
        """
        if sample_most_similar is None:
            prior = self.sim_to_other_classes # Doesn't make sense to alter this prior
            class_idx = np.random.choice(
                np.arange(self.sim_to_other_classes.shape[0]), p=prior
                )
        else:
            if sample_most_similar == 'first':
                class_idx = np.argmax(self.sim_to_other_classes) 
            elif sample_most_similar == 'second':
                class_idx = np.argsort(self.sim_to_other_classes)[-2]
            elif sample_most_similar == 'last': 
                class_idx = np.argmin(self.sim_to_other_classes)
            else:
                raise ValueError('Invalid sample_most_similar value')

        pval = self.sim_to_other_classes[class_idx]
        return class_idx, pval
    

    def sample_descriptive_property(self) -> str:
        """
        Sample a property (from the valid properties in the concept class).
        """
        prior = self.prior_over_properties['descriptive']
        prop_idx = np.random.choice(list(self.descriptive_properties), p=prior)
        pval = self.prior_over_properties['descriptive'][list(self.descriptive_properties).index(prop_idx)]
        return prop_idx, pval


    def sample_relative_property(self) -> str:
        """
        Sample a property (from the valid properties in the concept class).
        """
        prior = self.prior_over_properties['relative']
        prop_idx = np.random.choice(list(self.relative_properties), p=prior)
        pval = self.prior_over_properties['relative'][list(self.relative_properties).index(prop_idx)]
        return prop_idx, pval


    def sample_identifier_value(self, rotation_key=0) -> str:
        """
        Sample an identifier (non-PII or PII) from the valid properties in the concept class.
        """
        idproperty = np.random.choice(self.identifiers, p=self.prior_over_properties['identifiers'])
        pval = self.prior_over_properties['identifiers'][self.identifiers.index(idproperty)]
        id_val, p_idval, rotation_key = self.all_properties[idproperty].sample_descproperty_value(
            rotation_key=rotation_key,
            get_key=True
            )
        return id_val, pval * p_idval, rotation_key


