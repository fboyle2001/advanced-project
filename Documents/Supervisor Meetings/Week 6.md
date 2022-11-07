# End of Week 5
## Video
- Feedback

## Introduction
- Will smarten the diagrams up
- Worth adding some about the setup or would that be better placed at the beginning of a methodology or related work section?

## Results
- Fixed issues with data loading which has been a major problem
- Implemented Rainbow [drive images]
    - For each new task, separate the samples into their respective ground truth classes
    - Then for each class, sort the samples by the model’s uncertainty of the class of the sample
    - Uncertainty is calculated by applying transformations to the sample and using the model’s output on the prediction
    - Let k = (Max Memory Size) / (No. of Seen Classes)
    - Then for each uncertainty-sorted list of samples, select k samples:
        - Diverse: select k samples evenly spaced through the samples e.g. 0, 5, 10, 15...
        - Central: select the k least uncertain samples
        - Edge: select the k most uncertain samples
        - Random: select k random samples (without replacement)
    - Use these samples as the dataset to train the classifier
    - Default for Rainbow is Diverse
    - A bit slower than other techniques seen so far as it has to calculate uncertainty
    - Outperformed GDumb
- Started implementing Mnemonics
    - Another sampling technique rather than a complete solution
    - Will essentially strip the exemplar sampling from Rainbow and sub this method in to compare
    - Having a few difficulties but should be ok

## Timeline
- Starting to slip on the timeline
- Might reduce the number of techniques to implement: Mnemonics, L2P and maybe one other advanced technique?

## Next Week
- Finish Mnemonics
- Any final changes to the video
- Re-draft introduction