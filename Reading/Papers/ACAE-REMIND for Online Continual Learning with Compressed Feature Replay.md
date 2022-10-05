# ACAE-REMIND for Online Continual Learning with Compressed Feature Replay
## Skim Read
### Abstract
- An intermediate layer representation of samples is stored and leads to superior results and less memory usage
- Issue with existing methods trying this is that they used a fixed network backbone
- This paper proposes an Auxiliary Classifier Auto-Encoder (ACAE) which aims to solve this and has high compression rates to reduce memory usage
	- Allows saving of more images in the same space
- Tests on reasonably complex datasets

### Methodology
- Alternative to harder to train generative networks is feature replay
- Feature representations are ~50x smaller than storing the images directly
- Uses strong compression to achieve dimensionality reduction
- Train auto-encoder to create the feature representations?

### Results
- Comprehensive results on fairly complex datasets
- Memory used isn't always comparable
- Doesn't necessarily seem to convincingly outperform other methods though

### Further Action
- See cites 10, 21 about feature replay 
