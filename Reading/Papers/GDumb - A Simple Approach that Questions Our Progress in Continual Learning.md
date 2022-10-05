# GDumb - A Simple Approach that Questions Our Progress in Continual Learning
## Introduction
- Existing CL algorithms impose restrictive constraints that leave them impractical outside of their test scenarios
- These algorithms are extremely specific and thus of little practical relevance
- Also there is no explicit consensus on what the formulation of a CL algorithm involves leaving fractured experimental scenarios
	- None of them are representative of CL scenarios in the real world
- The paper proposes a simple algorithm with little simplifying assumptions called Greedy Sampler and Dumb Learner (GDumb)
- Basic outline:
	- Has a memory budget and the sampler stores samples from the data stream while making sure that the classes are balanced
	- At inference, the learner is trained from scratch on these stored samples
- Surprisingly, GDumb then provides SotA results by large margins in most cases despite not being designed to handle the intricacies of CL problems
- Raises signficant concerns related to popular assumptions and metrics

## Reformulating the CL Problem
- Wants to provide a realistic formulation of continual learning for classification
- Stream of training samples (or data accessible to the learner)
- Each sample is a two-tuple (x_t, y_t)
- Y_t = union of all seen labels at time t (so it is monotonically increasing)
- Objective is to provide f:x->y that accurately maps a sample x to a label y where y is in Y_t or y_hat indicating we don't recognise it
- y_hat is used to represent the incompleteness of the knowledge and notes that information might come from outside the training distribution
- Connects CL with the open-set classification problem
- The CL could improve its knowledge by learning semantics of samples inferred as y_hat
- Puts no constraintson the growing nature of the label space, nature of test samples and size of the output space
- No restrictions on resource consumption
- However, the lack of information about the nature and size of the output space has made the problem extremely hard
- Almost all of the work in this area imposes some additional constraints
- The constraints imposed across different works are non-standardised
- One common assumption is that the test samples always belong to the training distribution

## Common Assumptions
- Disjoint task formulation
	- Assume that at a particular point in time the data stream will provide samples specific to a task
	- This task order is often pre-determined. So the aim is to learn each task sequentially
	- Sharp transitions between tasks are the *task boundaries* 
	- This simplifies the general CL problem drastically as unknown nature of space is known
	- Provides a strong prior to the learner
- Task-incremental vs. Class-incremental
	- Extremely simplified form of continual learning
	- Information is passed by an oracle that determines the task/class and this information is then given to the model in a three-tuple (x, y, alpha)
	- Relatively impractical in real-world scenarios
- Online CL vs. Offline CL
	- Online CL allows the learner to use a sample only once unless it saves it to memory
	- Offline CL allows unrestricted access to the entire dataset for a particular task (not the previous ones) and samples can therefore be revisited over multiple passes
- Memory-based CL
	- We only have access to all/subset of samples corresponding to the current task - very restrictive and leads to catastrophic forgetting
	- Memory-based CL stores a subset of samples from each task to use while training the current task
	- Learner aims to generalise beyond current task, memoriser aims to remember a collection of memories from previous tasks. 
	
## Problems with Existing Algorithms
- Most algorithms and metrics inherently encode disjoint task assumption into their design which makes them difficult to generalise since any knock to this structure causes them to fail
- Task-incremental is very unrealistic with the use of an oracle
- Class-incremental is better but faces scaling issues since the label space is unrestricted
- Offline-CL and Online-CL are difficult to compare due to the trade off of space and time complexity

## GDumb
- Does not put any restrictions on the growing nature of the label space, task boundaries, online vs. offline and ordering of samples in the data stream
- Only requirement is to be allowed to stroe some episodic memories
- Does not claim to solve the general CL problem
- Surprisingly effective compared to other approaches
- Exposes shortcomings in existing algorithms
- Two key components: greedy balacing sampler and a learner
- Memory budget of k samples (store a maximum of k samples, is this per class??)
- Stores samples with aim to asymptotically balance class distribution
- Greedy in the sense that when a new class is encountered it creates a new bucket for the samples and starts removing samples from the old ones (particularly the one with the max number of samples)
- Assumes each stored sample is equally important
- Does not rely on task boundaries or information about number of samples in each class
- Objective of the learner (a DNN) is to learn a mapping from x to y for the samples stored in memory
	- Learns to classify all the labels seen until time t
- Makes a prediction using a softmax of probabilities over classes, a user defined mask and the Hadamard product
	- The mask lets it act like both a Task-incremental and Class-incremental learner
	- The mask is given at inference time
- This does not actually use y_hat that was just for theoretical consideration
- Since it does not impose any restrictions it does not require any hyperparameter tuning either
- Always trained in Class-incremental fashion with the option to mask to Task-incremental at evaluation time 

## Results
- Competitive and, in many cases, SotA results despite being a Class-incremental non-specific learner which is much more general and harder
- Performs exceptionally well on imbalanced data streams with blurry task boundaries
- Much more computationally efficient

## Future Extensions
- Active Sampling
	- Given importance value v_t in R<sup>+</sup> with a sample (x_t, y_t) at time t then we can store the most important samples instead
	- Quantifying importance is the difficult part
- Dynamic Probabilistic Masking
	- Extend masking beyond Class-incremental and Task-incremental to dynamic task hierarchies

## Conclusion
- Paper provides general overview of CL classification algorithms and their assumptions
- Proposes a simple, general algorithm GDumb
- Outperformed existing SotA in their own specific formulations despite not being tailored to the CL problem
- Aimed to provide a strong baseline to improve and benchmark against
- Raises questions about is CL really progressing in the right direction 

# Ideas
- Combine mnemonic storage memory with GDumb