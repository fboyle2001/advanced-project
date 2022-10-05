# Mnemonics Training - Multi-Class Incremental Learning without Forgetting
## Abstract and Introduction
- Keep around a few examples of previous concepts
- Effectiveness depends on how representatives these examples are
- This paper parameterises exemplars and makes them optimisable in an 'end-to-end' manner
- Network trained through bilevel optimisation (model-level and exemplar-level)
- Experiments on CIFAR-100, ImageNet-Subset and ImageNet
- Able to surpass SotA
- The automatically selected exemplars tend to be on the boundaries between classes
- Separation between classes weakens in later training phases

## Methodology
### Outline
- Uses an alternative to *herding* which was used in previous papers where they samples around the average of a class to find exemplars
- Uses an automatic exemplar extraction framework called *mnemonics* that parameterises exemplars using image-size parameters and then optimises them end-to-end
- This method is able to learn optimal exemplers for the new class while adjusting the exemplars of previous phases to fit the current data distribution
- Exemplars on boundaries of the class are needed to have a high-quality classifier
- Two models to optimise - conventional and parametised mnemonics
	- These 2 models are not independent and cannot be jointly optimised
	- Exemplars in current phase are used as input of later-phase models
- Optimises using a Bilevel Optimisation Program (BOP)
	- Alternates the learning of the two levels of models
- Perform a local BOP to distill knowledge of new class data into exemplars
	1) Temporary model is trained with exemplars as input
	2) Validation loss on ew class data is computed
	3) Gradients back-propagated to optimise input layer (parameters of mnemonics exemplars)
- These steps are iterated to derive representative exemplars for later training phases
- The exemplars can be regarded as synthesised rather than stored as they are optimised

### Training
- Classification model is incrementally trained in each phase on the union of new class data and old class mnemonic exemplars
- New class mnemonic exemplars are trained before omitting new class data
- In the global (meaning it operates through all phases) BOP the optimal model is used to optimise the exemplars and vice versa
- The number of exemplars << original data
- The exemplar method aims to ensure a feasible approximation of the assumption that the model trained on exemplars also minimises the loss on the original data
- Trains a temporary model on the exemplars to maximise predictions on the class that the exemplars were taken from
	- This is the local BOP problem
- To train the exemplars, we initialise parameters randomly as a sample subset S of the dataset and then train a temporary model (initialised based on the real model) and then train on the subset of data. Then back-prop the validation loss to the input layer.
	- Requires back-prop of a gradient through a gradient to compute Hessians
- Need to take care with classification bias as we have less exemplars than training data
- Weight Transfer Operations?? These come from 23, 32
	- Aim to preserve structural knowledge of the previous model while transferring networks to the next model

### Adjusting Exemplars
- Desirable to adjust the exemplars to the changing data distribution
- Old class data is unavailable
- Split the exemplars into two subsets
- Alternate between training with one and using the other to validate

## Implementation
- Two different approaches to memory. Either each class has a fixed number of exemplars or we have a fixed memory space so older classes can store more but will need to gradually delete some to free up space for new exemplars from new classes
	- Seems to come from citation 9
- See algorithm in the paper

## Conclusion and Results
- Benchmarking follows same method as LUCIR
- Model learns half of the classes and learns remaining classes evenly in subsequent phases
- Proposes a 'Forgetting Rate' which is different between accuracy of initial model and current model on the same initial test data. Lower = better
- Mnemonics is used as a plug-in module on other baselines replacing herding
- This method improves the results of existing baselines
- Selecting exemplars at the centre of classes appears deterimental as random sampling exemplars performs better
- The method proposed is generic and can therefore be used to improve other algorithms
- **Note**: When reading the results, upper bound is using all of the data in the dataset

## Further Reading
- Need to understand Weight Transfer Operations better
- Rebuffi et al. was first to define a protocol for evaluating MCIL methods
- Look at citations 2, 9, 16, 17, 25, 37
- Look at BOP cites 19, 29, 6, 34, 35
- Look at citations 2, 9, 25, 37 - MCIL papers
- Herding is defined in 36
- Look at distillation 17
- 11, 28, 33 use GANs to generate old samples
	- Results are very dependent on the GAN which is difficult to train
	- Also requires substantial amounts of memory
- Meta-learning is anothor type of BOP 5, 15, 31, 32, 38, 39
- **MCIL is proposed in 25** - distillation loss is shown in this (along with 17). Originally proposed in 8 but applied in 17, 25.
- Evaluation is done following 9, 25, 37
- Weight transfer 23, 32
- Uniform setting for the ResNet 9, 25