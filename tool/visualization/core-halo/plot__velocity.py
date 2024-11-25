import matplotlib
matplotlib.use('Agg')
import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt
from scipy import constants
from scipy.integrate import quad

# load the command-line parameters
parser = argparse.ArgumentParser( description='Profile' )

parser.add_argument( '-p', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )
parser.add_argument( '-halo', action='store', required=False, type=int, dest='halo',
                     help='which halo [%(default)d]', default=1 )

args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print (str(sys.argv[t]),end=' ')
#   print str(sys.argv[t]), 
print( '' )
print( '-------------------------------------------------------------------\n' )

idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix
halo        = args.halo

ds = yt.load('../Data_%06d'%idx_start)
### constant
kpc2km                = (1*yt.units.kpc).to('km').d
m_sun2_kg             = (1*yt.units.Msun).to('kg').d
hbar                  = constants.hbar *1e4*1e3 # cm^2 g/s
eV_c2                  = constants.eV/constants.c**2 *1e3 # g
if ds.cosmological_simulation:
    omega_M0          = ds.omega_matter
else:
    omega_M0          = 0.3158230904284232
omega_lambda          = ds.omega_lambda
h                     = ds.hubble_constant #dimensionless Hubble parameter 0.1km/(s*kpc)
hubble0               = h*0.1 # km/(s*kpc)
newton_G              = ds.units.newtons_constant.to('(kpc**3)/(s**2*Msun)').d  #(kpc^3)/(s^2*Msun)
background_density_0  = (1*ds.units.code_density).to("Msun/kpc**3").d
particle_mass         = (ds.parameters['ELBDM_Mass']*ds.units.code_mass).to('eV/c**2').d
code_mass             = (1*ds.units.code_mass).to("kg").d
CDM_particle_mass     = 170399.32174374 # Msun

### read data
df_FDM = pd.read_csv( 'Halo_Parameter_1' , sep = '\s+' , header = 0 , index_col='#')

FDM_v_path = ''

core_Both    = np.zeros(idx_end+1-idx_start)
core_QP      = np.zeros(idx_end+1-idx_start)
core_Bulk    = np.zeros(idx_end+1-idx_start)
inner_Both   = np.zeros(idx_end+1-idx_start)
inner_QP     = np.zeros(idx_end+1-idx_start)
inner_Bulk   = np.zeros(idx_end+1-idx_start)
halo_Both    = np.zeros(idx_end+1-idx_start)
halo_QP      = np.zeros(idx_end+1-idx_start)
halo_Bulk    = np.zeros(idx_end+1-idx_start)
average_Both = np.zeros(idx_end+1-idx_start)
average_QP   = np.zeros(idx_end+1-idx_start)
average_Bulk = np.zeros(idx_end+1-idx_start)

def shell_mass(r,dens):
    return 4*np.pi*r**2*dens

def CDM_dens(x, dens_parameter):
    r_CDM = dens_parameter[0]
    dens_CDM = dens_parameter[1]
    return 10**np.interp(np.log10(x), np.log10(r_CDM), np.log10(dens_CDM))

def potential_r(r,shell_mass,dens,dens_parameter,halo_radius,typ):
    def p(s):
        if s<r:
            return shell_mass(s,dens(s,dens_parameter))/r
        else:
            return shell_mass(s,dens(s,dens_parameter))/s
    if typ == 'small':
        potential_r, error = quad(p, 0, halo_radius)
    else:
        potential_r, error = quad(p, 0, halo_radius, epsrel = 0.01)
    return potential_r

def Jeans(r,dens,dens_parameter):
    enclose_mass,error = quad(lambda x:shell_mass(x,dens(r,dens_parameter)), 0,r)
    return -1*newton_G*enclose_mass/r**2*dens(r,dens_parameter)

def NFW_dens(r, dens_parameter):
    rho0 = dens_parameter[0]
    Rs = dens_parameter[1]
    return rho0/(r/Rs*(1+r/Rs)**2)

def soliton(x, core_radius):   
    return ((1.9*(2e-23/10**-23)**-2*(float(core_radius)**-4))/((1 + 9.1*10**-2*(x/float(core_radius))**2)**8))*10**9*m_sun2_kg*1000/kpc2km**3/1e15 #g/cm^3

def density(x, r, d):
    return 10**np.interp(np.log10(x), np.log10(r), np.log10(d))

def enclosed_mass(x, r, d):
    mass, error = quad( lambda a: shell_mass(a, density(a, r, d)), 0, x, epsrel=1e-2)
    return mass

def gforce_mult_dens(x, r, d):
    return enclosed_mass(x, r, d)/x**2*density(x, r, d)*newton_G

def jeans_v(x, r, d):
    inte, error = quad(lambda a:gforce_mult_dens(a, r, d), x, r[-1], epsrel=1e-2)
    return (inte/density(x, r, d))**0.5*kpc2km


count = 0
max_v = 0

def plot_v(path, suffix, name, idx, jeans =False, soliton_v =False):

    df_halo_parameter = pd.read_csv( path+'../../Halo_Parameter_1' , sep = '\s+' , header = 0 , index_col='#')
    current_time_z = 0
    current_time_a = 1
    halo_radius = df_halo_parameter['halo_radius'][idx]/current_time_a
    core_radius = df_halo_parameter['core_radius_1'][idx]/current_time_a

    ### load data
    df_gamer_dens         = pd.read_csv( path+'AveDens_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_bulk_r     = pd.read_csv( path+'AveVr_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_bulk_theta = pd.read_csv( path+'AveVtheta_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_bulk_phi   = pd.read_csv( path+'AveVphi_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_qp_r       = pd.read_csv( path+'AveWr_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_qp_theta   = pd.read_csv( path+'AveWtheta_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    df_gamer_v_qp_phi     = pd.read_csv( path+'AveWphi_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    # df_gamer_virial       = pd.read_csv( path+'VirSurf_%06d%s'%(idx,suffix) , sep = '\s+' , header = None ,skiprows =[0])
    
    ### read data
    gamer_dens            = df_gamer_dens[2] # rho_bg
    gamer_r               = df_gamer_dens[0]*1485 # kpccm
    gamer_shell_mass      = np.array(df_gamer_dens[6]) # code mass
 
    ### gamer v unit = 100 km/s v:bulk w:qp
    v_ave_r               = df_gamer_v_bulk_r[2]*100/current_time_a
    v_ave_theta           = df_gamer_v_bulk_theta[2]*100/current_time_a
    v_ave_phi             = df_gamer_v_bulk_phi[2]*100/current_time_a
    w_ave_r               = df_gamer_v_qp_r[2]*100/current_time_a
    w_ave_theta           = df_gamer_v_qp_theta[2]*100/current_time_a
    w_ave_phi             = df_gamer_v_qp_phi[2]*100/current_time_a
    v_sigma_r             = df_gamer_v_bulk_r[3]*100/current_time_a
    v_sigma_theta         = df_gamer_v_bulk_theta[3]*100/current_time_a
    v_sigma_phi           = df_gamer_v_bulk_phi[3]*100/current_time_a
    w_sigma_r             = df_gamer_v_qp_r[3]*100/current_time_a
    w_sigma_theta         = df_gamer_v_qp_theta[3]*100/current_time_a
    w_sigma_phi           = df_gamer_v_qp_phi[3]*100/current_time_a

    ### add hubble flow
    ### Recession Velocity = H(z)* distance, H^2(z) = H0^2(omega_M0(1+z)^3+omega_lambda)
    # hubble_flow           = (((hubble0)**2*(omega_M0*(1+current_time_z)**3+omega_lambda))**0.5)*(gamer_r*current_time_a)
    # v_ave_r               = v_ave_r + hubble_flow

    v_ave_2   = v_ave_r**2 + v_ave_theta**2 + v_ave_phi**2 + w_ave_r**2 + w_ave_theta**2 + w_ave_phi**2             # ave
    v_sigma_2 = v_sigma_r**2 + v_sigma_theta**2 + v_sigma_phi**2 + w_sigma_r**2 + w_sigma_theta**2 + w_sigma_phi**2 # sigma
    v2        = v_ave_2 + v_sigma_2                                                                                 # all
    v2bulk    = v_ave_r**2 + v_ave_theta**2 + v_ave_phi**2 + v_sigma_r**2 + v_sigma_theta**2 + v_sigma_phi**2       # bulk
    v2qp      = w_ave_r**2 + w_ave_theta**2 + w_ave_phi**2 + w_sigma_r**2 + w_sigma_theta**2 + w_sigma_phi**2       # qp

    ### soliton = 3.3 core_radius_1 (95% energy)
    soliton_range = (gamer_r <= core_radius*3.3)
    v2_core_both  = np.sum(v2[soliton_range]*gamer_shell_mass[soliton_range])/np.sum(gamer_shell_mass[soliton_range])
    v2_core_qp    = np.sum(v2qp[soliton_range]*gamer_shell_mass[soliton_range])/np.sum(gamer_shell_mass[soliton_range])
    v2_core_bulk  = np.sum(v2bulk[soliton_range]*gamer_shell_mass[soliton_range])/np.sum(gamer_shell_mass[soliton_range])

    ### inner = core_radius*6 ~ core_radius*8
    inner_range   = [core_radius*6, core_radius*8]
    filter_1      = (gamer_r >= inner_range[0])
    filter_2      = (gamer_r <= inner_range[1])
    v2_inner_both = np.sum(v2[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])
    v2_inner_qp   = np.sum(v2qp[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])
    v2_inner_bulk = np.sum(v2bulk[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])

    ### halo = core_radius*6 ~ halo_radius
    halo_range    = [core_radius*6, halo_radius]
    filter_1      = (gamer_r >= halo_range[0])
    filter_2      = (gamer_r <= halo_range[1])
    v2_halo_both  = np.sum(v2[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])
    v2_halo_qp    = np.sum(v2qp[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])
    v2_halo_bulk  = np.sum(v2bulk[filter_1 & filter_2]*gamer_shell_mass[filter_1 & filter_2])/np.sum(gamer_shell_mass[filter_1 & filter_2])

    ### ave = 0 ~ halo_radius
    all_range = (gamer_r<halo_radius)
    v2_ave_both = np.sum(v2[all_range]*gamer_shell_mass[all_range])/np.sum(gamer_shell_mass[all_range])
    v2_ave_QP   = np.sum(v2qp[all_range]*gamer_shell_mass[all_range])/np.sum(gamer_shell_mass[all_range])
    v2_ave_bulk = np.sum(v2bulk[all_range]*gamer_shell_mass[all_range])/np.sum(gamer_shell_mass[all_range])

    ### plot
    global count
    if count <3:
        plt.plot(gamer_r, v2**0.5, color = colors_tab10[count], lw = 2,label = 'v '+name)
        plt.plot(gamer_r, v2bulk**0.5*2**0.5, ':', color = colors_tab10[count*2+1], lw = 1,label = 'v bulk '+name)
        plt.plot(gamer_r, v2qp**0.5*2**0.5, '--', color = colors_tab10[count*2+2], lw = 1,label = 'v qp '+name)
    # else:
    #     plt.plot(gamer_r,v2**0.5, '.',ms = 6,mec = 'black',mew=0.5, mfc = colors_tab10[count-3], label = 'v '+name )
    # plt.plot([2e-1,4e2], [v2_core**0.5,v2_core**0.5], '--', color = colors_summer[count], lw = 1,label = 'v core '+name)
    # plt.plot([2e-1,4e2], [v2_inner**0.5,v2_inner**0.5], '--', color = colors_autumn[count], lw = 1,label = 'v inner '+name)
    # plt.plot([2e-1,4e2], [v2_ave**0.5,v2_ave**0.5], '--', color = colors_winter[count], lw = 1,label = '$<v_{halo}>$ '+name)
    
    # if count <1:
    #     # if count==0: plt.plot(gamer_r,v2bulk**0.5*2**0.5, lw = 3,color = colors_tab10[count], label = 'v bulk high')
    #     # plt.plot(gamer_r,v2qp**0.5*2**0.5,color = colors_tab10[count+1], label = 'v qp '+name)
    # else:
        # if count==1: plt.plot(gamer_r,v2bulk**0.5*2**0.5, '^',ms = 2,color = colors_tab10[count+1], label = 'v bulk low')
        # plt.plot(gamer_r,v2qp**0.5*2**0.5, '.',ms = 2,color = colors_tab10[count+2], label = 'v qp '+name)
    
    # plt.plot([2e-1,4e2], [v2_core**0.5,v2_core**0.5], '--', color = colors_winter[count], label = 'v core '+name)
    # plt.plot([2e-1,4e2], [v2_ave**0.5,v2_ave**0.5], '--', color = colors_summer[count], label = 'v ave '+name)
    # plt.plot([2e-1,4e2], [v2_inner**0.5,v2_inner**0.5], '--', color = colors_autumn[count], label = 'v inner '+name)

    if jeans:
        current_time_z = 0
        current_time_a = 1/(1+current_time_z)
        halo_radius_FDM = halo_radius

        df_dens = pd.read_csv(path+'../../prof_dens/Data_%06d_%d_profile_data'%(idx,halo), sep = '\s+' , header = 0 )
        dens = np.array(df_dens['density(Msun/kpccm**3)'])/current_time_a**3
        radius = np.array(df_dens['radius(kpccm)'])*current_time_a

        # # radius = np.logspace(-1, 3, num=100)
        # # dens = NFW_dens(radius,(1.6420655e+06, 2.4741994e+01))
        # sigma_jeans2 = np.zeros(len(radius))
        # potential = np.zeros(len(radius))
        # for i in range(len(radius)):
        #    potential[i] = newton_G*potential_r(radius[i],shell_mass,CDM_dens,(radius,dens),halo_radius_FDM,'general')
        #    partial_potential =  np.diff(potential) / np.diff(radius)
        #    r_half = 0.5*(radius[:-1]+radius[1:])
        # for i in range(len(radius)):
        #    sigma_jeans2[i], error = quad(lambda r:np.interp(r,r_half,partial_potential)*CDM_dens(r,(radius,dens)),halo_radius_FDM*3,radius[i],epsrel = 0.01)/CDM_dens(radius[i],(radius,dens))
        #    sigma_jeans2[i] = abs(sigma_jeans2[i])*kpc2km**2

        j_v = np.zeros(len(radius))
        for i in range(len(j_v)):
            j_v[i] = jeans_v(radius[i], radius, dens)

        # plt.plot(radius/current_time_a, sigma_jeans2**0.5*3**0.5, '-.',color = colors_tab10[count+8], lw = 1, label = 'v jeans')
        plt.plot(radius/current_time_a, j_v*3**0.5, '-.',color = colors_tab10[count+8], lw = 1, label = 'v jeans')

        if soliton_v:
            a = (2**(1.0/8) - 1)**(1.0/2)	
            core_mass_3 = ((4.2*10**9/((particle_mass/10**-23)**2*(float(core_radius)*10**3)))*(1/(a**2 + 1)**7)*(3465*a**13 + 23100*a**11 + 65373*a**9 + 101376*a**7 + 92323*a**5 + 48580*a**3 - 3465*a + 3465*(a**2 + 1)**7*np.arctan(a)))
            core_mass_1 = df_halo_parameter['core_mass_1'][idx]
            print(core_mass_3, core_mass_1)
            x = np.linspace(0.01,20,2000)*kpc2km*1e5  #cm
            dens = soliton(x/kpc2km/1e5,core_radius) #g/cm^3
            partial_dens_partial_x = np.diff(dens**0.5)/np.diff(x) #g/cm^4
            # x = 0.5*(x[1:]+x[:-1])
            dens = 0.5*(dens[1:]+dens[:-1])
            diff_x = np.diff(x)
            x = 0.5*(x[1:]+x[:-1])
            m = 4*np.pi*dens*(x)**2*diff_x # g
            Ek = 0.5*4*np.pi*partial_dens_partial_x**2*x**2*diff_x*(hbar/particle_mass/eV_c2)**2 #g cm^2/s^2
            Ek = Ek/1e10 #g km^2/s^2
            x = x/kpc2km/1e5/current_time_a #kpc
            v = (2*Ek/m)**0.5 #km/s
            plt.plot(x,v*2**0.5, '--',color = colors_tab10[count+6], lw = 1, label = 'v soliton')


    # if count ==0:
        plt.plot([halo_radius, halo_radius],[0,310], '--', lw = 0.5, label = 'halo radius FDM')
        plt.plot([core_radius, core_radius],[0,310], '--', lw = 0.5, label = 'core radius FDM')
        plt.plot([core_radius*3.3, core_radius*3.3],[0,310],'--',lw = 0.3,color = 'gold', label = '3.3rc')
        # plt.plot([core_radius*2, core_radius*2],[0,310],'--',lw = 0.3,color = 'gold', label = '2rc')
        # plt.plot([inner_range[0], inner_range[0]],[0,310],'--',lw = 0.3,color = 'gray')
        # plt.plot([inner_range[1], inner_range[1]],[0,310],'--',lw = 0.3,color = 'gray')
    count+=1

    core_Both[idx-idx_start]    = v2_core_both**0.5
    core_QP[idx-idx_start]      = v2_core_qp**0.5
    core_Bulk[idx-idx_start]    = v2_core_bulk**0.5
    inner_Both[idx-idx_start]   = v2_inner_both**0.5
    inner_QP[idx-idx_start]     = v2_inner_qp**0.5
    inner_Bulk[idx-idx_start]   = v2_inner_bulk**0.5
    halo_Both[idx-idx_start]    = v2_halo_both**0.5
    halo_QP[idx-idx_start]      = v2_halo_qp**0.5
    halo_Bulk[idx-idx_start]    = v2_halo_bulk**0.5
    average_Both[idx-idx_start] = v2_ave_both**0.5
    average_QP[idx-idx_start]   = v2_ave_QP**0.5
    average_Bulk[idx-idx_start] = v2_ave_bulk**0.5

    global max_v
    max_v = np.max(v2**0.5)


halo_vel_filename = 'halo_velocity_%d'%halo
writing_mode = 'append' if os.path.exists(halo_vel_filename) else 'new'

if writing_mode == 'new':
    with open( halo_vel_filename , 'w') as file:
        # writer = csv.writer(file, delimiter=' ')
        # writer.writerow(['#','a','rho0_FDM','Rs_FDM','Mass_NFW_fit_FDM','Radius_NFW_fit_FDM','Ep_NFW_fit_FDM',
        #                 'rho0_CDM','Rs_CDM','Mass_NFW_fit_CDM','Radius_NFW_fit_CDM','Ep_NFW_fit_CDM',
        #                 'Mass_FDM','Radius_FDM','Ep_FDM','Mass_CDM','Radius_CDM','Ep_CDM'])
        file.write(" #     time_a  core_Both     core_QP   core_Bulk  inner_Both    inner_QP  inner_Bulk   halo_Both    halo_QP    halo_Bulk average_Both     ave_QP    ave_Bulk\n")


writing_mode = 'append'

for idx in range(idx_start,idx_end+1,didx):

    group_n = 3
    colors_spring = plt.cm.spring(np.linspace(0, 1, group_n))
    colors_summer = plt.cm.summer(np.linspace(0, 1, group_n))
    colors_autumn = plt.cm.autumn(np.linspace(0, 1, group_n))
    colors_winter = plt.cm.winter(np.linspace(0, 1, group_n))
    colors_tab10 = plt.cm.tab10(np.linspace(0, 1, 10))

    plot_v('velocity/output_%d/'%halo, '', '', idx, jeans=True, soliton_v=True)
   

    plt.legend(loc = 'upper left', fontsize = 8)
    plt.xlim(2e-1,4e2)
    plt.ylim(0,np.ceil(max_v/50)*50)
    plt.xlabel('radius (kpccm)')
    plt.ylabel('$\sigma$ (km/s)')
    plt.xscale('log')
    plt.savefig('velocity/v2_of_radius/v2_of_r_%2d_%d.png'%(idx,halo), dpi = 150)
    plt.close()

    # v2_core_of_r = np.zeros(100)
    # r_2to4 = np.linspace(2,4,100)
    # for i in range(100):
    #     v2_core_of_r[i] = (np.sum(v2[gamer_r <= r_core*r_2to4[i]]*gamer_cell_mass[gamer_r <= r_core*r_2to4[i]])/np.sum(gamer_cell_mass[gamer_r <= r_core*r_2to4[i]]))**0.5
    # plt.plot(r_2to4,v2_core_of_r)
    # plt.xlabel('r_c')
    # plt.ylabel('v(km/s)')
    # plt.title('v_soliton vs r')
    # # plt.savefig('v2_core_of_r.png')
    # plt.close()
    
    

time_z = df_FDM['z'][:]
time_a = 1.0/(1.0+time_z)
time_a = time_a[list(range(idx_start,idx_end+1,didx))]

    
with open(halo_vel_filename , 'a') as file:
    writer = csv.writer(file, delimiter='\t')
    for i in range(idx_start,idx_end+1,didx):
        writer.writerow([i,'%.3e'%time_a[i], '%.5f'%core_Both[i-idx_start],'%.5f'%core_QP[i-idx_start],'%.5f'%core_Bulk[i-idx_start],\
            '%.5f'%inner_Both[i-idx_start], '%.5f'%inner_QP[i-idx_start], '%.5f'%inner_Bulk[i-idx_start],\
            '%.5f'%halo_Both[i-idx_start], '%.5f'%halo_QP[i-idx_start], '%.5f'%halo_Bulk[i-idx_start],\
            '%.5f'%average_Both[i-idx_start], '%.5f'%average_QP[i-idx_start], '%.5f'%average_Bulk[i-idx_start]])
