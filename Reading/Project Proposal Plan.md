# Project Proposal Plan
## RQs
- What are the most effective strategies to improve incremental learning performance in neural networks?
- How can multi-class incremental learning in neural networks be achieved?

## Lit Review
about 20-30 papers for lit review
skim all, read ~10 in depth
blackboard deadlines are correct
- critical comprehension
- evaluating papers, good and bad of them
- what's state of the art
- make sure to support arguments you make

Catastrophic Forgetting first described in French 1999, McCloskey and Cohen 1989, McClelland et al. 1995, Ratcliff 1990

Think one of the main things I need to do is set out the requirements for a successful lifelong learning algorithm, similar to (paper can't remember the name) and some assumptions about the data:
1) Data will arrive sequentially
2) It is possible that after some epochs N, we never see class K again but could still be expected to classify an item from class K in the future
3) ...

Do recent papers make much reference to the human brain or are they unguided and trying to guess what works?

## Planned Proposal Structure:
--- After having written a draft read the exemplar given so I can make changes
--- Probably want to standardise the language I use, am I referring to it as continual learning or incremental learning?

### Abstract
- Leave this to the end

### Introduction
- Cover what catastrophic forgetting is and what continual learning is 
- Why do we want to be able to train a neural network online
- Maybe touch on how this differs to reinforcement learning
- Applications: Image Classification --> OCR
- Defining what an algorithm that solves the continual learning problem should be capabale of (does this overlap too heavily with Task-IL and Class-IL definitions??)
- Maybe talk about the assumptions of normal neural networks and how incremental learning violates these and how continual learning looks to rectify them
- Talk about inspiration from the brain and neuroscience (or does this need to shift to related work I think this is fine in the introduction since related work is focusing on the techniques used in CS rather than Bio but this is nonetheless cruical to the narrative)
- Stability-plasticity dilemma

### Related Work
--- Perhaps need to add something about the pre-trained ImageNet model that is used for feature extraction? This is related to not wanting to focus on representation learning. This is *definitely* needed
--- Feature distillation?
--- Creating a narrative here is essential
--- Make sure to summarise -> analyse -> evaluate papers
--- Might need to watch https://youtu.be/t2d7y_r65HU

- Setting Types:
	- Class-IL vs Task-IL
	- Class-IL is more realistic and will be my focus
- Types of Solutions: Give examples of each, what they are and how they work
	- Memory Based (Rehearsal)
		- Storing exemplars
		- Feature representations
		- Generative models: both feature based and raw sample generation
	- Regularisation Based
		- EWC
		- Cosine Normalisation
	- Parameter Isolation Based
		- Can't remember name of method but there was a major one
	- Architectural Based
		- **NEED TO READ MORE FOR THIS** 
	- Provide a comparison at the end?
- Benchmarking
	- Baselines: finetuning (lower bound) and complete offline training (upper bound)
	- Computational efficiency, wall-clock time, metrics defined in other papers
- Datasets
	- Toy datasets: CIFAR10 and MNIST
	- Permutated datasets: in two minds about including this, maybe draft it and look to remove it afterwards
	- Complex datasets: CORe50, CIFAR-100, (Mini-)ImageNet
- Issues in the literature
	- Over complexity of solutions - see GDumb
	- Poor benchmarks and dataset selection
	- Trend towards Class-IL which is good
	- Need to think about identifying any gaps
- Conclusion of Lit Review
	- Take away messages -> memory-based seems to be most promising
	- How am I going to contribute (overlap with methodology?)
	- How am I going to draw on the related works?

### Methodology
- Think draft the related works section first and then pull ideas
- Essentially it is going to be training neural networks and analysing the results
- Need to setup a framework so that I can rigourously analyse results and make comparisons
- Talk about the different datasets used for evaluation as well
- RQ is experimental type
- Look at the types of data and data analysis I will be doing
	- Lectures 10-18 are good for this
- Will expand this much further once related works is drafted
- Experimental method
- Will primarily be focusing on memory-based methods?

### Validity
- Craig has made clear he is expecting quality work in the validity section
- Will be a primary focus to make sure this is high-quality then
- Use lectures, look at conference rankings, journal rankings
- How is my methodology valid => using benchmarks from the literature for comparability
- Will expand this much further once related works is drafted
- Using benchmarks defined in other papers
- Construct Validity - The benchmarks clearly measure retention of the information learned by the network
- Face Validity - Measuring information retention at the end clearly measures how well the network is retaining information that it has learnt previously
- Concurrent Validity - Using benchmarks and datasets from other papers
- Predictive Validity - Using multiple benchmarks and they should agree with each other