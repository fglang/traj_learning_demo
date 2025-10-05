# demo: learning-based k-space trajectory design
ESMRMB 2025 member session: Self-learning MR

click here to run in Colab:
https://githubtocolab.com/fglang/traj_learning_demo/blob/master/simple_SNOPY.ipynb

The demo is based on SNOPY (**S**tochastic optimization framework for 3D **NO**n-Cartesian sam**P**ling trajector**Y**) by Wang et al. (https://dx.doi.org/10.1002/mrm.29645) from the University of Michigan. \
Original code: https://github.com/guanhuaw/SNOPY

## Overview
There is a growing body of research on learning-based trajectory design, especially non-Cartesian and jointly learning a tailored NN-based reconstruction.

**Main objectives:**
- make efficient use of gradient hardware -> acquisition speed
    - comply with amplitude and slew rate constraints
    - peripheral nerve stimulation?
- favourable properties for reconstruction
    - parallel imaging:
        - avoid large gaps between samples (noise amplification -> CAIPI [Breuer])
        - ...but also not too close (multi-coil correlations -> Poisson disc [Vasanawala])
    - compressed sensing:
        - incoherent PSF
        - variable sampling density
    - deep learning reco
        - data-driven
        - tailored to specific anatomy?
- reduce susceptibility to artifacts
    - motion
    - off-resonance

**Challenges:**
- hardware imperfections (eddy currents, delays)
- many degrees of freedom
    - address by parametrization, e.g. splines
- training data
    - ideally complex multi-coil 



