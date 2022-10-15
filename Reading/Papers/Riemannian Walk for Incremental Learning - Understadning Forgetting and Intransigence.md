# Riemannian Walk for Incremental Learning - Understadning Forgetting and Intransigence
## Skim Read
### Abstract
* Introduces two metrics, forgetting and intransigence
* Proposes an algorithm called RWalk which is a generalisation of EWC++ and Path Integral
	* Theoretically grounded KL-divergence perspective
* Tested on MNIST and CIFAR-100

### Methodology
* Intransigence: Inability to update the knowledge to learn the new task
* Forgetting and Intransigence contradict and pose a trade-off
* Too much regularisation leads to intransigence, too little leads to forgetting
* Forgetting is difference between maximum knowledge gained about a task throughout the learning process and the knowledge the model currently has about it
* A few stored exemplars drastically reduces intransigence
* To measure intransigence, compare against a standard classification model which has access to all of the datasets at all times and compare to the actual model on a specific task
* RWalk has 3 key components:
	* a KL-divergence based regularisation over conditional likelihood
	* a parameter importance score based on the sensitivity of the loss over the movement on the Riemannian manifold
	* strategies to obtain a few representative samples from previous tasks
* First 2 parts handle forgetting, final part handles intransigence

### Results
* Not that much better than EWC++ really
* Uses CIFAR-100

### Further Action
