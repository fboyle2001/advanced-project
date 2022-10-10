# Dataset Distillation using Neural Feature Regression
## Skim Read
### Abstract#
* Aims to learn a small synthetic dataset that preserves as much information from the original dataset as possible
* Bi-level meta-learning 
* Addresses computation cost via Feature Regression with Pooling (FRePo)
	* Analogous to truncated backpropagation through time with a pool of models
* Uses CIFAR-100 and ImageNet datasets

### Methodology
* Knowledge distillation is a technique to compress knowledge
* Most previous work focused on model distillation where data is learnt by a large model and transferred to a smaller model
* Dataset distillation aims to learn a small synthetic dataset to preserve most of the information in the large dataset
* Aim is to accelerate model training and reduce costs
* Bi-level
	* Inner loop optimises model parameters on the distilled data (meta-parameters)
	* Outer loop refines the distilled data with meta-gradient updates
* Commonly use surrogate objectives as meta-gradient calculation on raw objectives too expensive but this has it's own problems
* Optimisation of the inner loop is where the main complexity is
	* Can simplify by training only the last layer of the neural network
	* Uses Kernel Ridge Regression
* Maintains a pool of models to prevent overfitting increases diversity
* Links to representation learning

### Results
* Doesn't really compare against any of the standard literature
* Generally pretty poor performance on the more complex datasets
* Decent enough on MNIST and CIFAR-10
* Faster training times and less memory used than similar models
* Has to train a model from scratch at meta evaluation step though? Is this ignored for the time results?

### Further Action
