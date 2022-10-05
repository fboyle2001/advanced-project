# iCaRL - Incremental Classifier and Representation Learning
## Skim Read
### Abstract
- Only requires training data for a small number of classes at the same time
- New classes can be added progressively
- Learns strong classifers and a data representation simulataneously
- Different from previous methods that were limited to fixed data representations and thus incompatible with deep learning architectures
- iCaRL is able to learn many classes incrementally over a long period of time where others fail

### Methodology
- Defines what it expects from a class-incremental learning algorithm:
	1) Trainable on stream of data where examples of different classes occur at different times 
	2) Should provide a competitive multi-class classifier for the classes observed so far
	3) Computational requirements and memory footprint must be bounded or grow very slowly w.r.t the number of classes seen so far
- iCaRL has 3 main components:
	- Nearest-mean-of-exemplars classification rule
	- Prioritised exemplar selection based on herding
	- Representation learning using knowledge distillation and prototype rehearsal
- For classification it dynamically selects a set of exemplar images out of the data stream for each class seen
- The total imaghes are limited by a parameter K to bound computational resources (K/t = number of examples in the t sets)
- Uses mean-of-exemplars classifier to classify images into the set of classes observed so far
- CNN under the hood
- Each time new classes are identified it updates its internal information
- Representation learning resembles ordinary network finetuning
- Also adjusts its exemplar sets each time it identifies a new class 

### Conclusion
- Very good results, improves on LwF, fixed rep, and finetuning
- Acknowledges performance is still lower than what systems achieve when trained in batches with all training data available 

### Further Action
