# Meta-Learning Representations for Continual Learning
## Skim Read
### Abstract
* Existing AI networks are not trained to facilitate future learning
* Proposes OML, an objective that directly minimises catastrophic interference by learning representations that accelerate future learning and are robust to forgetting
* Learn naturally sparse representations
* Complementary to existing methods
* Claims to be competitive with rehearsal-based methods

### Methodology
* NNs trained end-to-end are not good at minimising forgetting from single trajectory
	* Extremely sample-inefficient
	* Suffer catastrophic forgetting
* Two networks Representation Learning Network (RLN)  and Prediction Learning Network (PLN)
* Use a composition of the networks to consistute the model for the CLP
* Meta-training and training
* Learn a representation of the input and then use this to predict the class
* The RLN is updated in the outer loop (meta-training)
* The PLN is updated in the inner loop (meta-testing)
* Requires a meta-learning dataset and a representation benchmarking dataset?

### Results
* Unusual datasets
* Can be combined with existing methods to improve their performance, in some cases drastically
* Not combined with SotA but how is this paper?

### Further Action
