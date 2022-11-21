# End of Week 7
- Been a slow week

## Literature Review
- Struggling to figure out how to structure it
- Categorised each paper according to its setup
    - Online/Offline, Task-IL/Class-IL etc.
    - Plan is to have a table with all of the techniques grouped by their attributes
- Grouped all the techniques into a more finegrained structure
- What should I say about the papers?
    - Related work in the literature is quite short, sometimes just one sentence on a technique
    - Project prep were told to outline technique and conclusions, and add some critical evaluation
    - Is this good for the project or not?
    
## Implementation Write Up
- Drafted one for Rainbow
- Not entirely sure what to include
- Don't want to go into the results too much as will be putting in the evaluation instead
- Was thinking could include the experiments with different sampling strategies
    - Should help with flow rather than segregating into Part I and Part II
    
## Implementations
- Issue with Hindsight was it is Task-IL, oversight from me
- Learning to Prompt uses a pre-trained transformer, skeptical?
    - Does this defeat the point of online
    - Or is it valid still since the transformer is available online anyway?
    - Would of course contribute to the memory usage but that's ok can compare
- Determined going to implement:
    - ACAE-REMIND ?
    - ASER ? 