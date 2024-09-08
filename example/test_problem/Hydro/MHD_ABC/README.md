# Compilation flags
- Must enable
   - [[MODEL=HYDRO | Installation: Simulation-Options#MODEL]]
   - [[MHD | Installation: Simulation-Options#MHD]]
- Must disable
   - [[PARTICLE | Installation: Simulation-Options#PARTICLE]]
   - [[GRAVITY | Installation: Simulation-Options#GRAVITY]]
- Available options
   - [[Miscellaneous Options | Installation: Simulation-Options#miscellaneous-options]]


# Default setup
1. Periodic BC


# Note
1. Reference: [Zhang et al., 2018, ApJS, 236, 50](https://arxiv.org/abs/1804.03479)
2. `ABC_NPeriod` (n): 1 --> stable; 2 --> unstable
3. Initial perturbations have not been implemented yet
