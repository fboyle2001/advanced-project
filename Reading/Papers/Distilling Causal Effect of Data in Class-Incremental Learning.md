# Distilling Causal Effect of Data in Class-Incremental Learning
## Skim Read
### Abstract
- Causal framework to explain catastrophic forgetting
- Propose a method that supports existing techniques
- Forgetting happens due to the loss of causal effect of the old data in the new training
- Explains how existing techinques bring this causal effect back
- 'Distill' the Colliding Effect between old and new data - fundamentally equivalent to the causal effect of data replay without the cost of data storage
- Introduces the Incremental Momentum Effect of the data stream - removing this helps to retain the old data effect
- *Note*: it felt like the abstract threw a lot of terms out that didn't really offer that much insight into the paper compared to others I have read - avoid this style for the lit review

### Methodology
- Frames the data, feature and lable in each incremental learning step into causal graphs (which is a directed acyclic Bayesian graphical model)
		- (these are from Pearl 2000; Causality: Models, reasing, and inference 2nd edition)
- Use these graphs to formulate the causal effect between variables (causal intervention)
- Can calculate the causal effect (at least theoretically)
- Two independent varaibles becomes dependent upon knowing the value of their joint outcome i.e. the collider variable
- Moving averaged momentum in the optimiser bias towards the new data
- Propose a data effect distillation, didn't seem to understand much of it from the skim
- *AP:* Might be worth trying to figure out what the 'DDE' actually is here

### Conclusion
- Able to improve classification accuracy by augmenting existing state of the art (between 1-8% increases)
- The suggested method is a 'plug-in' to other algorithms
- Shows significant benefits in memory consumption though by not requiring a data replay
- The two methods are Distillation of Colliding Effect and Incremental Momentum Effect Removal
- Offers a way to save old data without actually storing it

### Further Action
- Interesting idea but performance benefits are massive
- Offers a method to save old data without storing it which is quite substantial
- Could be useful to tack on to a solution if I make one for some benefits
