# Generative Feature Replay For Class-Incremental Learning
## Skim Read
### Abstract
- Class-IL setting which is the hard version rather than Task-IL
- Image generation remains a hard problem and thus impractical
- Aim is to split the network into a feature extractor and a classifier
- Combines generative feature replay in the classifer with feature distillation in the feature extractor
- Reduces complexity of generative replay
- Claims approach is efficient and scalable to complex datasets
- Claims to achieve SotA on CIFAR-100 and ImageNet
- Uses fraction of storage compared to exemplar-based (i.e. using real samples) CL 

### Methodology
- Instead of generating images they propose generating features representing the images which is significantly easier
- Uses a hybrid generative feature replay at classifier level and distillation in the feature extractor
- Uses Canonical Correlation Analysis to analyse where the network is forgetting to improve insight
- Trains a feature generator which is then sampled from to produce feature representations of previously seen samples
- This can then be used to reinforce the knowledge learnt in previous steps

### Results
- Outperforms both exemplar and non-exemplar based method for most evaluated settings
- Shown that their method is efficient and scalable to large datasets
- Promising results which outperforms many methods
- Doesn't use the most recent methods, unclear if it came out before GDumb etc though
- Would be interesting to see how it fairs against GDumb
- LUCIR matches (and in some cases surpasses and vice versa) its performance on CIFAR-100

### Further Action
- Suggested extending to techniques from Semantic Drift Compensation for Class-Incremental Learning
- Was quite a solid paper