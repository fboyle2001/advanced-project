# Reinforced Continual Learning
## Skim Read
### Abstract
* Uses reinforcement learning to determine the best neural architecture for each coming task
* Uses CIFAR-100 dataset
* Claims to outperform existing approaches

### Methodology
* Adaptively expand the network by posing it as a combinatorial optimisation problem
* Uses a recurrent neural network determines parameters for each dynamic network (controller)
* Controller uses actor-critic strategy using validation accuracy and network complexity as rewards
* Consists of 3 networks: Controller, Value Network and Task Network
* Controller is a LSTM network for generating policies
* Value Network is a fully-connected network
* Task Network is any network of interest for solving the task e.g. a CNN for image classification
* Only trains the newly added nodes to the network to prevent drift
* Seems not to follow the complex Class-IL format and requires knowing the tasks?
* Seems to require knowing the task at inference time

### Results
* Uses low quality models or old models for comparisons
* Requires knowing the task at inference time
* Results dip dramatically when using CIFAR-100 compared to MNIST unsurprisingly
* Doesn't gain much over the compared models
* Takes a while to train - long wall-clock time

### Further Action
