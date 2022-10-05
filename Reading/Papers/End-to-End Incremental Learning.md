# End-to-End Incremental Learning
### Abstract and Introduction
- Uses a distillation loss to retain knowledge and a cross-entropy loss to learn new classes
- Learns the data representation and classifier jointly
- Evaluated on CIFAR-100 and ImageNet showing SotA performance
- Has a solid definition of what is expected from a continual learning algorithm 

### Methodology
- Has a representative memory that stores a small set of exemplars
- Any deep learning architecture can be used just replace the loss with their new incremental loss function
- Uses a deep network trained with a cross-distilled loss function (cross-entropy with distillation)
- When new classes are encountered, a subset of representative samples from them are selected and stored in the representative memory
- Two strategies for the memory:
	- Fixed memory size: Independent of number of classes so as # of classes increases, # of exemplars per class decreases
	- Fixed exemplars per class: Each class has a fixed K exemplars so the memory grows with the number of classes
- Uses herding selection to select new samples which produces a sorted list of samples of one class based on the distance to the mean sample of that class. Happens once per class when a new class is added to memory
- To remove samples, the samples are already stored in a sorted list as described above so only needs to remove samples from the end of the list
- Network consists of:
	- a feature extractor which transforms image into feature vector
	- a classification layer which is the last FC layer in the model, no. outputs = no. classes (replaced with softmax at test time)
- When new classes are trained we add a new classification layer corresponding to the classes and connect it to the feature extractor and the cross-distilled loss component
- Architecture of the feature extractor does not change during training - allows use of pre-trained models
- Uses cross-distilled loss:
	- Combines distillation loss (retains knowledge from old classes) with multi-class cross-entropy loss (learns to classify new classes)
	- Distillation loss is applied to the classification layers of old classes
	- Multi-class cross-entropy is used on all classification layers
	- Allows the model to update decision boundaries of the classes
	- Uses T = 2 for the distillation hyperparameter which gives remaining classes a greater influence which forces the network to learn a fine-grained separation between them
		- Learns a more discriminative model
	- Classification uses one-hot vectors as the label
	- Distillation uses logits produced by the classification layers on old classes as the labels (will have more than 1)
	- Samples from new classes are also used in distillation loss to produce gradients for both losses
- Features from the feature extractor will change during learning so the classification layers have to adapt to these new features
	- Differs from other approaches that only train the classification layers as they freeze the feature extraction layers
- Since we store a fixed number of samples => leads to data imbalance
	- Add additional fine-tuning stage with low learning rate and balanced subset of samples to deal with this
	- Adds a temporary distillation loss to avoid losing knowledge learned in the previous step
- After each incremental learning step it updates the memory
- In practice, models are implemented on MatConvNet. Uses dataset specific CNN models (layers differ but they are ResNets)

### Results
- CIFAR-100
	- Outperforms iCaRL and LwF
	- Quite a bit below the upper bound
	- Outperforms on both fixed memory size and fixed samples per class strategies
- ImageNet
	- Narrowly outperforms iCaRL
	- Best performance of compared methods
