# Online Class-Incremental Continual Learning with Adversarial Shapley Value
## Skim Read
### Abstract
- Memory replay techniques have shown promise
- This paper focuses on how to select which buffered images to replay and how to select which images to add to the buffer
- Contributes the novel Adversarial Shapley value scoring method which scores memory data samples according to their ability to preserve latent decision boundaries for previously observed classes
- Aim is to maintain learning stability and avoid forgetting 
- Competitive or improved performance on a variety of datasets

### Methodology
- In comparison to MIR, which seemingly selects samples redundantly, they select samples that are representative of different classes which are also near the boundary of each class 
- Single-headed approach
- *Shapley Value (SV)*: Origins in cooperative game theory, only allocation scheme satisifying group rationality, fairness, and additivity
- SV is used in machine learning to estimate individual contribution of data points in the context of all other data
- SV can be computed efficiently using KNN classifier since directly calculating it is O(2<sup>N</sup>) where as KNN is O(N log N)

### Results
- Slightly concerning that the authors don't use GDumb as a baseline here
- They present their results without highlighting when other algorithms beat theirs which is concerning
- Results are somewhat promising, presentation does undermine them however
