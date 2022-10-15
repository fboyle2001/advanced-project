# Efficient Lifelong Learning with A-GEM
## Skim Read
### Abstract
* Averaged GEM
* Does a comparison of existing methods
* Proposes a Task-IL formation?

### Methodology
* Suggest metric called Learning Curve Area
* Rather than ensure the loss of individual previous tasks does not increase (like in GEM), A-GEM tries to ensure that the average episodic memory loss over the previous tasks does not increase
* Reduces the $t-1$ constraints in GEM to $1$ constraint in A-GEM
	* Better computation efficiency
* GEM has better guarantees than A-GEM does in terms of worst case forgetting
* Suggests use of Joint Embedding Model with Compositional Task Descriptors to improve all techniques

### Results
* Not too complex datasets, paper is 2018 though
* Much faster wall-clock time than GEM 
* Accuracy and forgetting are similar for A-GEM and GEM
* Neither are that revolutionary nor are they compared against good techniques

### Further Action
