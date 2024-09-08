# Compilation flags
Enable  : MODEL=HYDRO, SRHD, NCOMP_PASSIVE_USER = 4
Disable : GRAVITY, MHD, COMOVING, PARTICLE
- Must enable
   - [[MODEL=HYDRO | Installation: Simulation-Options#MODEL]]
   - [[SRHD | Installation: Simulation-Options#SRHD]]
   - [[NCOMP_PASSIVE_USER=4 | Installation: Simulation-Options#NCOMP_PASSIVE_USER]]
- Must disable
   - [[GRAVITY | Installation: Simulation-Options#GRAVITY]]
   - [[COMOVING | Installation: Simulation-Options#COMOVING]]
   - [[PARTICLE | Installation: Simulation-Options#PARTICLE]]
   - [[MHD | Installation: Simulation-Options#MHD]]
- Available options
   - [[Miscellaneous Options | Installation: Simulation-Options#miscellaneous-options]]


# Default setup
1. Units:

   [L] = 5 kpc
   [M] = 1.0e5 Msun
   [V] = 1 c
   -->[T] = [L]/[V] ~ 16.308 kyr
   -->[D] = [M]/[L]^3 ~ 5.41e-29 g/cm**3

2. Default resolution ~ 39 pc ([[MAX_LEVEL | Runtime-Parameters:-Refinement#MAX_LEVEL]]= 3)


# Note
1. ZuHone et al. 2023, Galaxies, vol. 11, issue 2, p. 51
