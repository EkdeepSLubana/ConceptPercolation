# A Percolation Model of Emergence

Codebase for the paper ["A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language"](https://arxiv.org/) 


## Requirements 

To set up the requirements, install conda and use the following command.

`conda env create -f conceptPercolation.yml`


## Example execution 

```
python train.py data.n_descriptive_properties=460
```

This variable maps to number of properties as follows: `40 * (data.n_descriptive_properties - 10)`


All relevant variables to execute experiments can be found in `config/conf.yaml`.