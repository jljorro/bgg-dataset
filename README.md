# Boardgames Dataset RecSys 2025

Repository that contains recommender models evaluations for new dataset from BoardGameGeek.

## Installation

These scripts requires:

- Python 3.10
- CUDA 12.5

### Install dependencies

```bash
pip install -r requirement.txt
```

### Scripts

TODO

### Recommender models tested

- *General Recommender Models*
  - Random
  - Pop
  - ItemKNN
  - UserKNN

- *Context-Aware Recommender Models:*
  - ContextRandom
  - ContextPop
  - FM
  - DeepFM

### Folders

- data: data corresponding to the dataset described in the paper, divided in ratings, continuous, and discrete, the last two with the context extracted either from metadata or reviews
- elliot: configuration files and code to use with Elliot
- evaluation: scripts to evaluate the models
- prefiltering: new implemented prefiltering methods
- recbole: configuration files and code to use with RecBole


## License

This dataset is intended for research and educational purposes. Please ensure compliance with data source licenses when using or distributing derivative works.
