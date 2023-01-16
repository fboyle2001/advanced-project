# Week 12 Meeting
## Write Up Review
- Definitely needs some work

## Novel Implementation
- Having some doubts about different bits
- Should I focus fully on the uncertainty measurement instead?
    - It does give a performance boost 
    - Is this not just finetuning the ViT though?
    - Sufficiently novel enough as it really just builds on a lot of techniques
    - Worth doing some experiments around this to show it works in addition to the main comparison?
    - Look at the distribution of uncertainty in each class?
    - Propose an additional method to compare, take distance to closest and second closest then take ratio
    - Could further generalise to use n closest as a hyperparameter?
- ViT with no training using NCM classification and Random Sampling: ~70%
- ViT with an MLP head using NCM classification and Random Sampling with CE Loss: ~70% accuracy
- ViT with an MLP head using NCM classification and Random Sampling with SCL Loss: ~70% accuracy
- ViT with an MLP head using NCM classification and Uncertainty Sampling with SCL Loss: ~72% accuracy
- Can enhance these further by using LR warming 
- Should I write up about these experiments in the novel implementation section and then lead up to the final algorithm?

## Result Generation
- Currently generating results during the week
- Expect to be ready to start writing the results and conclusions up towards the end of the week
- Produced some example graphs, after some feedback
    - Wall-clock time, better to just have a table instead?
    - Stacked or grouped multi-bar chart?
    
## This Week
- Finish results generation
- Clean up the code for the demo
- Start writing the results section
