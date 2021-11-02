import argparse
import sys
import yt
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

# load the command-line parameters
parser = argparse.ArgumentParser( description='Plot profile and out put Halo_parameter' )

parser.add_argument( '-p', action='store', required=False, type=str, dest='prefix',
                     help='prefix [%(default)s]', default='../' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )

args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print str(sys.argv[t]),
print( '' )
print( '-------------------------------------------------------------------\n' )


idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix

yt.enable_parallelism()

# DM background density 3H^2/(8 \pi G)*\Omega_m
omega_M0         = 0.2835         
hubble0          = 6.9550000e-02   #km/(s*kpc) 
newton_G         = 4.3*10**-6     #(kpc*km^2)/(s^2*Msun)
background_density_0 = (3*hubble0**2*omega_M0)/(8*math.pi*newton_G)

# plot variable
nbin             = 64

ts = yt.load( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )

# change x, y, z to the guessed halo center
coordinates = ts[0].arr( [ 1.32565917969, 0.0223876953125, 0.300610351562 ] , 'code_length' ) 

number = int(idx_start)

for ds in ts.piter():
    # periodicity
    ds.periodicity = (True, True, True)

    # find center 
    # find the maximum value in a sphere extending halo_radius from center_guess 
    halo_radius  = 0.1            # halo radius in Mpc/h --> change this value properly
    center_guess = coordinates
    sphere_guess = ds.sphere( center_guess, (halo_radius, 'code_length' ) )
    center_find   = sphere_guess.quantities.max_location( 'density' )
    center_coordinate  = ds.arr( [center_find[1].d, center_find[2].d, center_find[3].d], 'cm' )
    # use this coordinate rerun to guess the center until it not change
    print(center_coordinate.in_units('code_length'))  

    # extract halo
    sp  = ds.sphere( center_coordinate, (halo_radius, 'code_length') )

    prof_mass_accumulate = yt.create_profile( sp, 'radius', fields= 'cell_mass',
                                              weight_field = None, n_bins = nbin ,
                                              units={'radius': 'kpc', 'cell_mass': 'Msun'}, 
                                              extrema = {'radius': (7e-1,1e3)}, accumulation = True)

    prof_dens = yt.create_profile( sp, 'radius', fields= 'density',
                                   weight_field = 'cell_volume', n_bins = nbin ,
                                   units={'radius': 'kpc','density': 'Msun/kpc**3'}, 
                                   extrema = {'radius': (7e-1,1e3)})
    
    radius_o = prof_dens.x.value
    density_o = prof_dens['density'].value                       # density at radius 
    mass_accumulate_o = prof_mass_accumulate['cell_mass'].value  # all mass within radius

    # remove zero
    radius = []
    density = []
    mass_accumulate = []

    for i in range(len(radius_o)):
        if(density_o[i]!=0):
            radius.append(radius_o[i])
            density.append(density_o[i])
            mass_accumulate.append(mass_accumulate_o[i])

    radius = np.array(radius)
    density = np.array(density)
    mass_accumulate = np.array(mass_accumulate)
    
    # calculate virial mass to know halo radius
    current_time_a = ds.current_time
    current_time_z = (1.0/float(current_time_a)) - 1.0

    omega_M = (omega_M0*(1 + current_time_z)**3)/(omega_M0*(1 + current_time_z)**3 + (1 - omega_M0))
    zeta = (18*math.pi**2 + 82*(omega_M - 1) - 39*(omega_M - 1)**2)/omega_M 
    zeta_0 = (18*math.pi**2 + 82*(omega_M0 - 1) - 39*(omega_M0 - 1)**2)/omega_M0

    # Mmin
    eV = 1.6021766208e-12
    c = 2.99792458e10
    particle_mass = ds.parameters['ELBDM_Mass']*ds.parameters['Unit_M']/(eV/c**2)
    Mmin = 4.4*10**7*(particle_mass / 10**-22)**(-3/2.0)

    # use mass_accumulation directly
    for i in range(1,len(radius)):
        virial_mass2 = zeta*background_density_0*(radius[i]**3*4*math.pi/3)
        if (mass_accumulate[i] <= virial_mass2):
            virial_mass1 = zeta*background_density_0*(radius[i-1]**3*4*math.pi/3)
            halo_radius = ((virial_mass2-mass_accumulate[i])*radius[i-1] + (mass_accumulate[i-1]-virial_mass1)*radius[i] ) / (virial_mass2-mass_accumulate[i]+mass_accumulate[i-1]-virial_mass1)
            halo_mass = ((virial_mass2-mass_accumulate[i])*mass_accumulate[i-1] + (mass_accumulate[i-1]-virial_mass1)*mass_accumulate[i] ) / (virial_mass2-mass_accumulate[i]+mass_accumulate[i-1]-virial_mass1)
            break               
    
    # # devide mass_accumulation by volume
    density_all = []
    for i in range(len(radius)):
   	    density_all.append(float(mass_accumulate[i]) / float(radius[i]**3*4*math.pi/3) - zeta*background_density_0)

    for k in range(len(radius)):
        if density_all[k] >= 0 and density_all[k+1] <= 0:
            halo_radius2 = (radius[k]*abs(density_all[k+1]) + radius[k+1]*abs(density_all[k])) / (abs(density_all[k]) + abs(density_all[k+1]))
            halo_mass2 = (mass_accumulate[k]*abs(density_all[k+1]) + mass_accumulate[k+1]*abs(density_all[k])) / (abs(density_all[k]) + abs(density_all[k+1]))  
        
    # core radius 1 : x = xc  use for loop
    # core radius 2 : xc = max/2
    # core radius 3 : x = xc solve equation 

    # core radius 1
    peak_density = []
    for j in range(len(radius)):
	    peak_density.append(((1.9*(float(current_time_a)**-1)*(particle_mass/10**-23)**-2*(float(radius[j])**-4))*10**9) - (float(max(density))))

    for h in range(len(radius)):
        if peak_density[h] >= 0 and peak_density[h+1] <= 0:
            core_radius_1 = (radius[h]*abs(peak_density[h+1]) + radius[h+1]*abs(peak_density[h])) / (abs(peak_density[h]) + abs(peak_density[h+1]))
            #print('core radius 1' , core_radius_1*current_time_a)   
            a = (2**(1.0/8) - 1)**(1.0/2)	   
            core_mass_1 = ((4.2*10**9/((particle_mass/10**-23)**2*(float(core_radius_1*current_time_a)*10**3)))*(1/(a**2 + 1)**7)*(3465*a**13 + 23100*a**11 + 65373*a**9 + 101376*a**7 + 92323*a**5 + 48580*a**3 - 3465*a + 3465*(a**2 + 1)**7*math.atan(a)))
            #print('core mass 1' , core_mass_1)   

     #core radius 2
    for m in range(len(radius)-1):
        if density[m] != 0 and density[m+1] != 0:
            if (density[m] - max(density)/2) >= 0 and (density[m+1] - max(density)/2) <= 0:  
                core_radius_2 = (radius[m]*abs(density[m+1] - max(density)/2) + radius[m+1]*abs(density[m] - max(density)/2)) / (abs(density[m] - max(density)/2) + abs(density[m+1] - max(density)/2))
                core_mass_2 = (mass_accumulate[m]*abs(density[m+1] - max(density)/2) + mass_accumulate[m+1]*abs(density[m] - max(density)/2)) / (abs(density[m] - max(density)/2) + abs(density[m+1] - max(density)/2))
                #print('core radius 2' , core_radius_2*current_time_a)   
                #print('core mass 2'   , core_mass_2)
                #print('core mass 2_chmr'   , (core_mass_2*current_time_a**0.5)/Mmin)
    
    #core radius 3
    core_radius_3 = (max(density)/10**9/1.9*float(current_time_a)*(particle_mass/10**-23)**2)**-0.25
    a = (2**(1.0/8) - 1)**(1.0/2)	
    core_mass_3 = ((4.2*10**9/((particle_mass/10**-23)**2*(float(core_radius_3*current_time_a)*10**3)))*(1/(a**2 + 1)**7)*(3465*a**13 + 23100*a**11 + 65373*a**9 + 101376*a**7 + 92323*a**5 + 48580*a**3 - 3465*a + 3465*(a**2 + 1)**7*math.atan(a)))
            
    
    # collecting halo propertities (in comoving coordinate)
    with open("Halo_Parameter_" , 'a') as file:
        writer = csv.writer(file)
        x = str(number)
        a = str("{:.1e}".format(particle_mass))
        b = str((float(center_coordinate.in_units('code_length')[0])))
        c = str((float(center_coordinate.in_units('code_length')[1])))
        d = str((float(center_coordinate.in_units('code_length')[2])))
        e = str("{:.4e}".format(float(current_time_z)))
        f = str("{:.7e}".format(float(halo_radius*current_time_a)))
        g = str("{:.7e}".format(float(halo_mass)))
        h = str("{:.7e}".format(float(halo_radius2*current_time_a)))
        i = str("{:.7e}".format(float(halo_mass2)))
        j = str("{:.7e}".format(float(max(density)*current_time_a**-3)))
        k = str("{:.7e}".format(float(core_radius_1*current_time_a)))
        l = str("{:.7e}".format(float(core_radius_2*current_time_a)))
        m = str("{:.7e}".format(float(core_radius_3*current_time_a)))
        n = str("{:.7e}".format(float(core_mass_1)))
        o = str("{:.7e}".format(float(core_mass_2)))
        p = str("{:.7e}".format(float(core_mass_3)))
        q = str("{:.5e}".format(float((halo_mass2*(zeta/zeta_0)**0.5)/Mmin)))
        r = str("{:.5e}".format(float((core_mass_2*current_time_a**0.5)/Mmin)))

        writer.writerow([' '.join([x,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r])])
    
    # plot profile
    def soliton(x):   
        return ((1.9*(float(current_time_a)**-1)*(particle_mass/10**-23)**-2*(float(core_radius_1)**-4))/((1 + 9.1*10**-2*(x/float(core_radius_1))**2)**8))*10**9
    
    anal = soliton(radius)
    plt.plot( radius, density, 'bo', label='simulation' )
    plt.plot( radius, anal, label='analytical' )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-1,1e3)
    plt.ylim(1e-1,1e8)
    plt.ylabel('$\\rho(r)/\\rho_{m0}$')
    plt.xlabel('radius(kpc/a)')
    plt.legend(loc = 'upper right')
    plt.title('z = %.2e core radius = %.2f kpc/a'%(current_time_z, core_radius_1*current_time_a), fontsize=12) 
    FileOut = 'fig_profile_density_%06d_o' %number+'.png'
    plt.savefig( FileOut)
    plt.close()

    number += int(didx)

