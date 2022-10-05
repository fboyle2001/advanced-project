# Online continual learning in image classification: An empirical survey
- Compares state of the art methods
- Maximally Interfered Retrieval (MIR), iCaRL and GDumb primarily
- GDumb provides strong baseline
- MIR produces performance near to offline training
- Survey focuses on Online Continual Learning where not all of the previous data remains available
- Also considers Domain Incremental (where the data is non-stationary) and Class Incremental
- Uses Single-Headed which is good since multi-headed tends to be unrealistic
- Only considers supervised image classification
- iCaRL is competitive when competing on small replay buffers
- GDumb is a strong baseline outperforming many methods in medium-size datasets
- MIR performs best in large datasets
- GDumb struggles in Domain Incremental
- Online-able methods are able to do a single pass of the data (bar anything stored in their memory)
- Has a section on comparable metrics/measures
- Another issue is that a large proportion of CL work tunes hyperparameters by using the whole dataset (offline) which makes the results too ideal and unrealistic

## Techinques
- Regularisation: Impose constraints on the update of network parameters such as by imposing additional penalty terms into loss functions or modifying the gradient of parameters during optimisation
- Knowledge Distillation: 
- Memory: Stores a subset of samples from previous tasks for replay or regularisation
- Generative Replay: Trains a deep generative model to generate fake data in place of a memory (disadvantage is that they are complex and expensive to train)
- Parameter Isolation: Allocate different parameters to each task. Can be subdivided into Fixed Architecture (only activates relevant parameters for each task without changing network) and Dynamic Architecture (adds new parameters for new tasks)

## Compared Methods
### Regularisation-Based
- EWC / EWC++ (++ uses a soft update of the Fisher Information Matrix)
- LwF

### Memory-Based
- Averaged GEM (A-GEM) uses parameters constraining
- iCaRL
- Experience Replay
- Maximally Interfered Retrieval
- Gradient Based Sample Selection
- GDumb

### Parameter Isolation-Based
- Continual Neural Dirichlet Process Mixture - dynamic architecture method

## Tricks in Class-IL setting
- Class imbalance in memory-based techniques remains a problem
- Class imbalance in general is one of the most crucial causes of catastrophic forgetting
- Class imbalance leads to bias in the fully connected layer
- **Labels Trick**: Only consider the outputs for the classes in the current mini-batch when calculating cross-entropy loss. Prevents overpenalising classes not in the mini-batch.
- **Knowledge Distillation with Classification Loss (KDC)**: KD is an effective way to transfer knowledge between networks. Introduces loss based on KD and a parameter lambda, suggests an improvement KDC* that uses an adaptive lambda that is better
- **Multiple Iterations (MI)**: Perform multiple gradient updates per mini-batch 
- **Nearest Class Mean (NCM) Classifier**: Takes the biased FC layer. Uses (approximate) prototype vectors to assign class labels instead
- **Separated Softmax (SS)**: Use one softmax for new classes and one for old classes to avoid overly penalising old classes.
- **Review Trick (RV)**: An additional fine-tuning step with a small learning rate using balanced subset from memory and training set of current task

## Datasets
- Split CIFAR-100
- Split MiniImageNet
- CORe50-NC - designed for class incremental learning

## Experiments
### Regularisation Techniques
- **EWC++**: Almost same performance as the lower bound. Exploding regularisation gradient is identified as root cause.
  - Huber regularisation could be a potential solution
- **LwF**: Relies on KD, similar performance to memory-based in Split CIFAR-100 and Split Mini-ImageNet. Fails on CORe50-NC which is a more realistic dataset.

### Memory-based Methods
- **A-GEM**: completely fails
- **iCaRL**: shows best performance with small memory buffer in Split CIFAR-100 and Split Mini-ImageNet. Other methods fail (such as ER, MIR, GSS since replaying with a small memory buffer yields severe class imbalance), 
- **GDumb**: with a large memory buffer, GDumb outperforms other methods by a large margin. Confirms concerns about process in the literature. GDumb needs increasingly high amounts of memory to be effective. Requires a significant amount of running time. Begins to slip on CoRE since it requires lot of memory to perform well.
- **MIR**: outperforms GDumb on the larger datasets as it is more robust to varying memory size
- Other methods are ER and GSS. GSS claims to be an enhanced version of ER but is outperformed by ER consistently.

### Parameter Isolation Methods
- **CN-DPM**: shows competitive results in Split CIFAR-100 and Split Mini-ImageNet but fails on CORe5-NC. Likely due to the sensitivity to hyperparameters. Has low levels of forgetting compared to similar accuracies. Requires a significant amount of running time though.

## Experiments on Tricks
- RV was used to bring A-GEM performance up suggesting indirect use of memories is ineffective
- For ER and MIR, LB and NCM were most effective at small memory sizes (leads to doubling in accuracy)
- KDC, KDC* and SS boost accuracy on average around 60-70%
- At larger memory sizes, NCM remains effective, RV becomes much more useful
- KDC fails at higher memory size => over-regularisation on KD term
- KDC* has much better performance than KDC
- MI and RV are sensitive to memory size as they are highly dependent on the memory
- *Paper Concern*: Doesn't apply these tricks to GDumb despite it being the most effective?
- Summary: NCM is good across all memory sizes, LB is better at small memory sizes, RV is better at large memory sizes
- Runtimes: NCM and RV scale with increase in memory size, others are fixed overhead

## Possible Areas of Exploration
- All methods have a strong bias towards new classes by the end of the training. Maybe look at ways to improve this?
- Suggests that weights for new classes are higher than for old classes - could look at some 'sleep' equivalent to fix the network?
- Raw-Data-Free Methods: storing raw samples is not feasible due to privacy and security concerns
- Generative Replays: one direction but impractical for complex datasets
- Feature Replays: promising direction where latent features of old samples at given layer (feature extraction layer) are replayed rather than raw data
- Can combine generative and feature replays
- SDC, DMC, DSLDA, InstAParam, EBM-CL might be worth a look
- Meta-learning: MER, OML, iTAML, La-MAML, MERLIN, Continual-MAML
- Areas other than image classification
- RNNs and LSTMs also suffer from CF
- Real-world scenarios such as chat bots, self-driving cars and in clinical settings

## Discussion
- Previous works found that even with a small memory buffer (so high class imbalance), catastrophic forgetting wasn't a major problem in multi-headed approach. Suggests the issue lies with the FC layer being biased rather than the feature extractor
- Gradient projection approach is ineffective as evidenced by A-GEM. Direct use of memories is better.
- For memory-free LwF is effective on CIFAR-100 and Mini-ImageNet but they all fail on CORe50-NC
- For small memory buffers, iCaRL is best in CIFAR-100 (slightly) and Mini-ImageNet (by a lot) followed by CN-DPM
- For large memory buffers, GDumb outperforms specially designed methods for the CL problem in CIFAR100 and Mini-ImageNet at the cost of high training times
- In the realistic CORe50-NC dataset, MIR is consistently the best across memory sizes
- Confirm experimentally and theoretically that a key cause of CF is bias towards new classes in the last fully connected layer due to data imbalance
- No method showed any positive forward or backwards transfer due to bias
- For small memory buffers LB and NCM tricks were best
- For large memory buffers, NCM and RV are best
- CL is getting better but not there yet