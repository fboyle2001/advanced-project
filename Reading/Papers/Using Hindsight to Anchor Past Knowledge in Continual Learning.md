## Using Hindsight to Anchor Past Knowledge in Continual Learning
### Abstract
* Experience replay is popular
* Complements experience replay with new objective called anchoring
* Bilevel optimisation
* Anchor points are learned using gradient based optimisation to maximise forgetting
* Continual learning would allow moving past the iid assumption

### Methodology
* Seems certain to be online
* HAL leverages bilevel optimisation to regularise the training objective with one representational point per class per task, called anchors
* Anchors constructed via gradient ascent in the image space by maximising one approximation to the forgetting loss for the current task throughout the entire continual learning experience
* Estimate the amount of forgetting that the learner would suffer on these anchors if it were to be trained on future tasks in hindsight
* That is by measuring forgetting on a temporary predictor that has been fine-tuned on the episodic memory of past tasks
* Anchors lie close to the classifier's decision boundary
* Two-step parameter update rule
* First step computes a temporary parameter vector by minimising loss at a minibatch and the episodic memory (the usual experience replay parameter update)
* Second step employs a nested optimisation to perform the actual update of the parameters theta, trades off:
	* the minimisation of the loss value on the minibatch and episodic memory
	* and changhes in predications at the anchor points for all past tasks
* Also has mathematical underpinning
* Proposes an optimisation objective consisting of forgetting loss and mean embedding loss
* Tries to push the anchor point embedding towards the mean data embedding
* Meaning embeddings are computed as running averages and is initially 0
* Learn one exemplar per class for each task

### Results
* Not compared against the best papers
* Claims SotA
* Uses sufficient datasets: CIFAR-100 and Mini-ImageNet

### Further Action
