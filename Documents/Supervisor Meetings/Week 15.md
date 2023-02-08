# Week 15 Meeting
## Paper Template
* Intro: slightly under 3 pages
* RW: 4.5 pages
* Methodology+Existing+Novel: slightly under 5 pages
* Results+Evaluation: 4.5 pages
* Conclusion: To do, have ~1.5 pages to work with though
* Expect conclusion to be between 0.5 and 1 page leaving a small amount of space to add stuff when redrafting
* What is the difference between the results and evaluation? Do I need to split

## CIFAR-10 5k
* First task forgetting useful?
* Top-1 accuracy instead of Top-5 accuracy, is this inconsistency ok?

## CIFAR-10 0.5k
* SCR outperforms GDumb in this low memory scenario in contrast to the 5000 setup
* DER++ outperforms Rainbow
* Worth continuing to generate results if already at page limit?

## Analyse of L2P forgetting
* Can see one of the flaws in L2P is that it doesn't necessary learn much about the shared information between tasks until Task 5
* At this point it is effectively overwriting and is consistent with the drop off in performance seen in Figure 6b
* Prompt 2 and Prompt 10 were the most used in Task 1 and then again in Task 5 which is consistent with a halving in the classification accuracy of Task 1 after Task 5 has been used for training