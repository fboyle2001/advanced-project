# Gradient based sample selection for online continual learning
## Skim Read
### Abstract
- Formulates the sample selection as a constraint reduction problem based on a constrained optimisation view of continual learning
- Idea is to maximise the diversity of samples available in the buffer
- Shows comparable or better results when compared to existing methods

### Methodology
- Formalised as a solid angle minimisation problem
- Propose a surrogate objective for the minimisation problem
- Uses a greedy algorithm to sample from the buffer - as efficient as reservoir sampling without the imbalanced data stream
- Requires no i.i.d assumptions
- Goal is to optimise the loss on the current example without increasing the loss on the previous examples
- Solving the constraints increases linearly with the number of previous samples and hence is unbounded
- Instead approximate with a fixed computation and storage budget (i.e. max number of storable samples)
- Solve the constraints around the feasible region
- Use a surrogate objective in order to decrease the number of constraints and reduce the feasible set size

### Results
- Doesn't use the most complex of datasets which undermines it slightly
- Outperforms reservoir sampling
- Outperforms iid online on Disjoint CIFAR-10
- Not the most comprehensive study of the results unfortunately
	- Lacking complex datasets
	- Lacking strong baselines such as GDumb

### Conclusion
- Aimed to diverse samples in the gradient space 
- Advantageous when task boundaries are blurry or data is imbalanced
