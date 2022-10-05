# Rainbow Memory - Continual Learning With a Memory of Diverse Samples
## Skim Read
### Abstract
- Focuses on the harder Class-IL scenario
- Argues the importance of diversity in the samples in an episodic memory
- Proposes a new memory management strategy which is based on per-sample classification uncertainity and data augmentation called *Rainbow Memory*
- Test on MNIST, CIFAR10, CIFAR100, and ImageNet
- Shows it significantly improves the accuracy in the Class-IL setup 

### Methodology
- Many new sampling methods only show marginal gains compared to uniform random sampling from memory
- This paper focuses on increasing the quality of the stored samples rather than increasing sample efficiency
- Samples stored in the episodic memory should be representative of their class and discriminative of other classes
- Samples near the centre of class distributions are representative and samples at the boundary are discriminative 
- Propose to store samples that are *diverse* in the feature space
- Need some way to estimate the location of samples in the feature space. Expensive to do so
- Proposes to estimate the relative location by uncertainty of a sample estimated by the classification model 
- Samples are sorted in memory by their uncertainty
- Use data augmentation to further diversify the samples - not sure this is really unique but rather a common pre-processing technique

### Results
- Tested on good datasets: MNIST, CIFAR10, CIFAR100, and ImageNet
- Compared against quality SotA: EWC, Rwalk, iCaRL, BiC, and GDumb
- Outperforms the other methods especially at low memory sizes suggesting that the sampling strategy is effective

### Further Action
- Might be worth reading their Related Works section as this is quite a recent paper compared to some of the others
