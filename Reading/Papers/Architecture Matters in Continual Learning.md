# Architecture Matters in Continual Learning
## Skim Read
### Abstract
* Most literature is concerned with algorithmic part of continual learning on fixed architecture
* Paper shows choice of architecture can significantly impact the continual learning performance
* Vanilla fine-tuning with modified components can achieve similar or better performance than specifically designed CL methods on a base architecture without significant difference in parameter count

### Methodology
* Argues the inductive biases induced by architectural components are important for continual learning
* Removal of Global Average Pooling layer in ResNet-18 makes finetuning very competitive
* Suggests architecture and algorithmic solutions can be complimentary and lead to improvements in results
* Carries out a comparison across CNNs, MLPs, ResNet-D, WideResNet-D-N and ViT architectures
* Records average accuracy, learning accuracy, joint-task accuracy and average forgetting
	* See paper for definitions §2.1.2
* ResNet-18 and ResNet-34 have roughly the same performance despite RN-34 having ~2x the parameters
	* Similar applies to WRN but less parameters seems to be better
* Wide neural networks forget less catastrophically (Mirazadeh et al 2021)
* Scaling models by width has more impact than depth scaling

### Results
* ResNet and WRN have higher learning accuracy implying they are better at learning new tasks
* CNNs and ViTs are better at retaining information (lower forgetting score)
* Average forgetting of CNNs is smaller than other architectures
* Simple CNNs achieve the best trade-off between learning and retention
* Impact of Batch Normalisation is data-dependent
	* If the input distribution is stable they help, otherwise they hurt
* Skip connections do not seem to have an impact on continual learning (positively or negatively)
* Average Pooling does not have any significant impact
* Max Pooling improves the learning capability of the network significantly
* Global Pooling Layers contribute to narrowing of the network leading to increased forgetting
	* Not inherent issue with GAP layers but rather the narrowing of the network is the problem
	* ResNet can improved in CL settings by removing GAP layers
* Increasing Attention Heads in ViTs does not improve CL performance efficiently
* ViTs are robust against distributional shifts
* For CNNs recommends adding Batch Normalisation and Max Pooling
* For ResNets recommend removing GAP layers or locally averaging the features in the penultimate layer
	* Increases knowledge retention
* Combining a good CL algorithm with a good CL architecture leads to best results

### Further Action
