# FearNet: Brain-Inspired Model for Incremental Learning
## Skim Read
### Abstract
- Incremental learning violates assumptions in normal neural networks
- This paper proposes using a generative model which does not store samples (compared to iCaRL which is SotA when this paper released in 2017/18)
- Uses a dual-memory system inspired by the brain in which memories are consolidated from recent memories to a long-term storage network
- Inspired by mechanisms that occur during sleep
- SotA on CIFAR-100 and CUB-200

### Methodology
- 3 neural networks:
	- One for short-term storage
	- One for long-term storage
	- One to determine which network to use
- Uses a generative autoencoder for pseudo-rehearsal
- Two complementary memory centers
- Short-term memory learns new information for recent recall
- During sleep phases, FearNet uses a generative model to consolidate data from the short-term network to the long-term network (intrinsic replay)
- Uses another network that predicts the probability hat the long-term memory contains the class or whether the short-term does
- The selection network is trained separately => might be a bit of a problem!

### Results
- Uses metrics from Kemker et al. 2018
- CIFAR-100, CUB-200 and AudioSet datasets are used
- Results are excellent and excels on the complex datasets
- Uses much less memory than storing exemplars
- Outperforms existing methods

### Further Action
- Might be worth a deeper read to get a feel for how they do this exactly and it is similar to what I was thinking but could be improved upon
- Gillies 1991 observed that unsupervised models are more robust to CF due to the lack of target outputs to forget
- Assumed Gaussian model could be better to use Gaussian Mixture Model to get improvements
- Paper suggests the following improvements:
	- Integrate selection network into the model directly
	- Use a semi-parametric model for the short-term memory
	- Learn feature embeddings from raw inputs
	- Replace pseudorehearsal mechanism with a generative model which doesn't require storing class statistics 