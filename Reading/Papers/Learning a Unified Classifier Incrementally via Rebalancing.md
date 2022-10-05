# Learning a Unified Classifier Incrementally via Rebalancing
## Skim Read
### Abstract
- DNNs conventionally trained offline and require large dataset prepared beforehand
- Reveals the imbalance between previous and new data is a crucial cause of catastrophic forgetting
- Unified classifier that treats both old and new classes uniformly
- Incorparates cosine normalisation, less-forget constraint and inter-class separation

### Methodology
- Imbalances:
	- Imbalanced magnitudes: magnitude of weight vectors of new classes are significantly higher than those of old classes
	- Deviation: relationship between the features and the weight vectors of old classes are not well preserved
	- Ambiguities: weight vectors of new classes are close to those of old classes
	- The combined effect of the these 3 effects can severely mislead the classifier causing biased decisions towards the new classes 
- Fixing Imbalances:
	- Cosine Normalisation: enforces balanced magnitudes across all classes
	- Less-forget Constraint: aims to preserve the geometric configuration of old classes
	- Inter-class separation: encourages a large margin to separate old and new classes
	- All this occurs during the training process

### Conclusion
- Outperforms iCaRL by a large margin on CIFAR100
- Brings consistent improvements under different settings

### Further Action
- Seems a promising paper, could certainly be worth a further read as seems SotA.