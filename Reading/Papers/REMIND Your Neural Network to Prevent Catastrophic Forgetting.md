# REMIND Your Neural Network to Prevent Catastrophic Forgetting
## Skim Read
### Abstract
- Evidence from neuroscience suggests that the brain replays compressed memories but existing replay based methods use raw memories instead
- Paper proposes REMIND which uses compressed representations
- Outperforms other methods for incremental learning on ImageNet ILSVRC-2012 and CORe50

### Methodology
- Hippocampus Indexing Theory indicates that the brain stores compressed representations of neocortical activity while awake
- Consolidates memory by playing these patterns back
- REMIND: Replay using Memory INDexing
- Stores hidden representations using tensor quantisation (CNN features)
- Compression is implemented using Product Quantisation
- Two-step process:
	1) Compress current input
	2) Reconstruct a subset of previously compressed and mix in with the current input and then update the plastic weights of the network using these
- Compressing the images allow REMIND to store far more in the same space
- Uses PQ to compress
- Have a fixed memory budget
- Randomly remove examples from the class with the most examples when we have a full memory budget

### Results
- Uses Top-5 accuracy as a measure
- Outpeforms others on ImageNet and CORe50
- Fairly promising