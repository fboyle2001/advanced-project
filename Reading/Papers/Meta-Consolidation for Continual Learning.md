# Meta-Consolidation for Continual Learning
## Skim Read
### Abstract
* Presents MERLIN 
* Assumes weights of a NN for solving a task t come from a meta-distribution that is learned incrementally
* Uses CIFAR-100 and Mini-ImageNet
* Claims to beat baseline methods and SotA

### Methodology
* Proposes a different perspective to address continual learning based on the latent space of a weight-generating process rather than weights themselves
* Uses a variational auto-encoder (VAE)
* Trains base network
* Learns the parameter distribution
* Samples parameters from distribution to form ensemble of models used to predict at evaluation time
* Train a set B of base models on random subsets of incoming data
* Use this set of models to learn the parameter distribution for the task using VAEs
* Sample model parameters for all tasks seen so far and use the decoder of the VAE to refine the overall VAE (meta-consolidation)

### Results
* Results aren't amazing
* Compared against other meta-learning and not really top models
* Makes it hard to compare again

### Further Action
