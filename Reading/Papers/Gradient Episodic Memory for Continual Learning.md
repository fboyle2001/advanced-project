# Gradient Episodic Memory for Continual Learning
## Skim Read
### Abstract
* Proposes set of metrics to evaluate models
* Also tests on ability to transfer knowledge
* Proposes Gradient Episodic Memory (GEM)
* Uses MNIST and Incremental CIFAR-100
* Early memory-based paper
* Discusses flaws in Empirical Risk Minimisation (ERM) Vapnik, 1998 

### Methodology
* Defines Backward Transfer as influence that learning task t has on previous tasks k < t
* Defines Forward Transfer as influence that learning task t has on future tasks k > t
* Stores a subset of observed samples in an episodic memory
* Task-IL formation?

### Results
* Incremental CIFAR-100
* Old paper so not much to compare
* Outperformed regularisation approaches (EWC)

### Further Action
