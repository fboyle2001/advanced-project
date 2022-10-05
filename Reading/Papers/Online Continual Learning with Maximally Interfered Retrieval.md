# Online Continual Learning with Maximally Interfered Retrieval
## Skim Read
### Abstract
- Replay methods have been shown to be effective
- Proposes a controlled sampling strategy for memories in the replay
- Retrieves the samples that are most *interfered* meaning those whose prediction will be most negatively impacted by the foreseen parameters update
- Shows this produces consistent gains in performance and reduced forgetting
- Includes source code

### Methodology
- Considers the online continual setting where samples are seen once and are not iid. This is the more realistic setting
- This is a replay based method rather than prior-focused regularisation as they have been shown to be ineffective on long sequences of data
- Aim is to learn a model that doesn't interfere with previous performance
- Similar to other replay methods just a change in sampling strategy
- Estimate the would-be parameters from the current batch
- Compute the change in loss between the new and old loss for samples in the memory
	- An alternative strategy is to also store the minimum loss seen so far for each sample instead
- Uses random sampling to first select some samples to rank from memory to improve efficiency
- Consider both storing samples in memory and using a generative model trained to produce samples instead
- Introduces a hybrid approach too using an autoencoder to compress the data stream and simply the MIR search

### Results
- Storing memories: Tests on CIFAR-10 which is a bit basic, outperforms regular episode-replay using random sampling
- Generative replay: Find it is not a viable strategy on CIFAR-10 yet
- Results aren't excellently presented but the empirical survey paper shows MIR is good on more complex datasets too

### Conclusion
- Interfered sampling reduces forgetting and improves on random sampling
- Indicates generative modelling is not yet good enough
- Shown that using encoded memories is feasible 

