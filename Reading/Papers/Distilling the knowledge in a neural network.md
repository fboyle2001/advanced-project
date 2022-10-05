# Distilling the knowledge in a neural network
## Skim Read
### Abstract
- Ensemble models are an easy way to improve performance
- Can be too computationally expensive and cumbersome though
- Previous work has shown that ensemble models canbe compressed into a single model
- This paper develops this approach further using a different compression technique
- Distills the knowledge in an ensemble of models into a single model
- Also introduces a new type of ensemble using one or more full models and many specialist models that distingush fine-grained classes which would otherwise confuse full models
- These specialist models canbe trained rapidly and in parallel
- *Distillation* is the process of reducing the large ensemble into a single smaller model

### Methodology
- Idea is that a large ensemble generalises well so if we can distill this knowledge into a smaller model then it will outperform a small model trained on the training data which does not generalise well
- Their more general solution is to raise the temperature of the final softmax so that the cumbersome model is able to produce a suitably soft set of targets. This temperature is then used to train the small model to match these soft targets
- This temperature, T, is not simply scaling the distribution but affects the probabilities (loosely probabilties) --> see the softmax equation

### Results
- The distilled models are much easier to deploy than the cumbersome ensembles
- Also shows that performance of a really big network can be improved by learning a large number of specialist networks that learn to differentiate between classes in tight clusters
	- Doesn't show how to distill knowledge from specialists back into the single large net though, acknowleges this

### Further Action
- If going to make use of distillation go and read Model Compression, Caruana et al. 2006.
