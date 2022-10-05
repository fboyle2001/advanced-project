# Insights from the Future for Continual Learning
## Skim Read
### Abstract
- Proposes prescient continual learning where we incorporate existing information about classes before doing any training
- Adds future classes with no training samples
- Introduces *Ghost Model* which is a representation-learning model using ideas from zero-shot learning
- Generative model of representation space along with careful adjustment of losses allows exploiting of insights from future classes to constraint spatial arrangements
- Uses the AwA2 and aP&Y datasets?? 

### Methodology
- Will be able to use *ghost* features which are predicted features for future classes which makes room in the representation space for future classes
- Is task-IL which is the easier situation where the dataset is split into tasks based on label
- Proposed Ghost Model consists of a convolutional feature extractor, a feature generator, and a classifier
- Generator generates ghost features based on future classes it hasn't seen yet
- The classifier is then trained on these ghost features as well
- Prior information about the classes needs to be provided so that ghost features can be generated
- Inspired by zero-shot learning where the model needs to make predictions on unseen classes

### Results
- Relies heavily on having access to attribute data for the unseen classes
- Results are on datasets with attributes available which make them difficult to compare to other methodss seen so far
- On MNIST it appears to have better accuracy than a baseline but it is task-IL so it is not really well suited


