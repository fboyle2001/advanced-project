# Learning to Prompt for Continual Learning
## Skim Read
### Abstract
* More succinct memory system than rehersal-based 
* Introduces prompts which are small learnable parameters maintained in memory space
* Optimises the prompts to instruct prediction and manage task-invariant and task-specific knowledge while maintaining plasticity

### Methodology
* Draws on prompt-based learning which is a transfer learning technique
* These prompts instruct the model to perform tasks conditionally
* Leaves the pre-trained model untouched and instead learns a set of prompts to instruct the model dynamically to solve corresponding tasks
* The query mechanism to lookup prompts is known as the Prompt Pool - explicitly decouples shared and task-specific knowledge
* Class-IL focus and task-agnostic setting
* High-level idea is to apply a function to modify the input so that the model gets additional information about the task
* Ideally learn prompts to share knowledge between similar tasks while maintaining knowledge independence when necessary
* Task identity is unknown at test time so task-independent prompts are not feasible
* Key-value pair based query strategy to dynamically select suitable prompts for the input
* Introduce a determinsitic query function w.r.t to the different tasks with no learnable parameters
* Extract features using the pre-trained model
* Prompts are prepended using a transformer encoder with the extracted features of the input
* This amalgamation is then fed into the classifier
* Have a rehersal-based improvement for competing against rehersal-based methods

### Results
* Compared against EWC, LwF, ER, GDumb, BiC, DER++ and Co<sup>2</sup>L
* Outperformed all of them and fairly close to the upper bound even without the rehersal strategy
* Uses CIFAR-100 and CORe50 which are suitable datasets
* Average accurcy and forgetting - good choices
* Includes an abalation study

### Further Action
