# Generative vs Discriminative - Rethinking the Meta-Continual Learning
## Skim Read
### Abstract
* Leverage meta-learning to encourage the model to learn continually
* Develop a generative classifier that efficiently uses data-driven experience to learn new concepts
* Claims to be immune from forgetting even with few samples
* Claims negligible computational overhead using a generative model

### Methodology
* 'Meta-learning is an endeavor to gather experience from solving related problems in order to solve new unseen ones more efficiently'
* Meta-learn an embedding space 
* Episodic meta-learning
* Proposes a probabilistic generative method that is 'immune to forgetting'
* Has a feature-extractor network that embed input into representation space
* 'Concept learning'
* Generative Bayesian Classifier instead of a discriminative one
* Hyper-parameters of the generative model are computed via meta-training
* Not too sure how this all works?? 
### Results
* Doesn't compare against memory-based methods?
* Uses sufficiently difficult datasets
* No offline-training or finetuning bounds
* Just compares with existing meta-learning methods

### Further Action
