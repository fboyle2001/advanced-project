# Elastic Weight Consolidation
## Abstract and Introduction
- Can overcome the catastrophic forgetting limitation by selectively slowing down learning rates of specific weights that are important for tasks
- Prior approaches ensure all tasks are simultaneously available during training - 'multitask learning paradigm'
- Tasks presented sequentially can be stored in replay buffers, 'system-level consolidation', which is impractical for learning large numbers of tasks
- Evidence from nature suggests continual learning in animals relies on task-specific synaptic consolidation, makes these neurons less plastic

## EWC
- EWC attempts to replicate this in an algorithm for continual lifelong learning
- Many configurations of weights will result in the same performance
- Over-parameterisation increases the probability of finding a solution of task B centred around the optimal parameters for another task A
  - Use a quadratic penalty - like a spring
  - Stiffness of the penalty should vary by parameter
  - Should be stiffer for parameters that are important for task A
- Posterior probability is intractable so approximate as a Gaussian with mean given by the parameters of the task and diagonal precision given by the diagonal of the Fisher information matrix
- For more than two tasks, we enforce multiple quadratic penalities - difficulty to optimise increases => lower performance expected?
- In SGD, forgetting compounds as you train further
- It is important to penalise specific weights in the network since by penalising all of the weights equally it leaves little capacity to train on any future tasks
- Dropout regularisation does not scale well to more tasks making it unsuitable (and its performance is worse than EWC)
- Has an analysis of how the network is dividing the available weights up e.g. are they overlapping or is the network being subdivided. Either way they share more weights the closer they get to the output layer => shared output classes
- Bayesian NNs could be used to perhaps improve performance

### Applying to reinforcement learning
- Use EWC in Deep Q Networks
- EWV has fixed resources rather than increasing the capacity to add additional tasks
- RL tasks provide a further difficulty in that the agent needs to infer which task is currently being performed, when to switch tasks, and when a new task is being learnt
- Task context is used as the latent variable in a Hidden Markov Model
- Allow some pre-training by opting not to apply an EWC penalty until 20 million frames

## Conclusion
- EWC consolidates knowledge of previous tasks to protect them during learning
- Does this by selectively decreasing the plasticity of the weights => similar to neurobiological observations
- For EWC to be semi-fast it requires simplifications and assumptions such as parameterising by a Gaussian and using the diagonal of the Fisher Information matrix
- Each synapse the weight, variance and mean of itself
