# Semantic Drift Compensation for Class-Incremental Learning
### Abstract
- Studies incremental learning for embedding networks
- Proposes a method to estimate the drift(?) called semantic drift 
- Experiments on CIFAR-100 and ImageNet-Subset
- Claims to have competitive results and can combine with existing methods to improve results

### Methodology
- Takes a different approach by aiming to estimate drift rather than aiming to prevent forgetting directly
- Uses an embedding network with a metric learning loss rather than a classification loss
- Task-agnostic (algorithm has no access to task label at test time so this is the harder better scenario)
- Embedding networks map data into a low-dimensional output where distance represents the semantic dissimilarity
- ENs simultaneously perform feature extraction and metric learning
- In embedding space can apply L2-distance to determine similarity between originals
- Mentions Siamese networks
- Conventional appracoh is softmax classifier with cross-entropy loss
	- Problems are that outputs are heavily tied to prediction classes => difficult when new classes are introduced, requires new neurons
	- Leads to bias towards new classes
- Embedding networks have advantages such as being able to accomedate new classes without structual change
- EN uses NCM for classification (from iCaRL), conventionals use max probability (softmax)
- Softmax outperforms initially but drops off as more classes added
- Adapts existing methods to use embeddings networks

### Semantic Drift
- Embeddings suffer from drift when learned sequentially
- Proposes a drift compensation 
- Once drift is calculated can attempt to compensate for it in the EN instead

### Results
- Presents fairly solid results
- Competitive with iCaRL and beats non-exemplar methods
- Uses CUB-200, CIFAR-100, ImageNet-Subset and Caltech-101
- Would be interesting to compare to newer methods too
- Idea of embedding network seems effective

### Further Action
