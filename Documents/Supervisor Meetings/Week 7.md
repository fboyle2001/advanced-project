# End of Week 6

## Introduction
- Will smarten the diagrams up
- Worth adding some about the setup or would that be better placed at the beginning of a methodology or related work section?

## Implementation
- Been struggling with implementing techniques this week 
- Realised there was an issue in the code with Rainbow but results still hold
- Experimenting with Rainbow sampling, Beta distributions
- Proportional

## Results
- Cat class always struggles even on Offline training
- Compare planes and cat classes on edge vs central
- Edge:
    - Samples not uniform
    - Background variation (e.g. Bird, Deer. Frog, Plane) esp compared to central
    - Non-uniform orientations
    - Planes are non-standardised types (i.e. not passenger jets) planes performs poorly like cats
- Central:
    - Uniformity in colour (e.g. Truck, Car, backgrounds related to edge - lots of red)
    - Less obscurity (e.g. see Planes)
    - Pretty standard poses for animals compared to edge

## Next Week
- Going to try and get this technique implemented
- Aiming to make a good start on some of the bits of the paper as I'm away this weekend
- Remaining techniques: MIR (meta-learning) and Learning to Prompt although skeptical 