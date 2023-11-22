import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import yt

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
   print(sys.argv[t], end = ' ')
print( '' )
print( '-------------------------------------------------------------------\n' )


idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx
prefix      = args.prefix
halo        = args.halo

# get background_density_0
ds = yt.load('../'+prefix+'Data_0000%d'%idx_start)
background_density_0 = (1*ds.units.code_density).to("Msun/kpc**3").d

compare_path = '/projectZ/vivi235711/Gadget2/L_2.8_N_256_seed_705107/plot_script/'

df_Halo_Parameter = pd.read_csv( 'Halo_Parameter_%d'%halo , sep = '\s+' , header = 0 , index_col='#')


def plot_profile(path,name, core_is_true = True):

    def soliton(x):   
        return ((1.9*(float(current_time_a)**-1)*(particle_mass/10**-23)**-2*(float(core_radius_1)**-4))/((1 + 9.1*10**-2*(x/float(core_radius_1))**2)**8))*10**9/background_density_0

    # read data
    df = pd.read_csv( path+'/prof_dens/Data_%06d_%d_profile_data'%(idx,halo) , sep = '\t' , header = 0 )
    df_halo_parameter = pd.read_csv( path+'/Halo_Parameter_%d'%halo , sep = '\s+' , header = 0 , index_col='#')

    current_time_z = df_halo_parameter['time'][idx]
    current_time_a = 1/(current_time_z+1)
    radius = df['radius(kpccm)'][:]
    density = df['density(Msun/kpccm**3)'][:]
    dens = density/background_density_0
    halo_radius = df_halo_parameter['halo_radius'][idx]/current_time_a
    # plot
    plt.plot( radius, dens, '.', label=name)
    # plt.plot( [halo_radius, halo_radius], [1e-1, 1e9], '--', label= name+' halo radius')
    # if FDM
    if (core_is_true):
        particle_mass = df_halo_parameter['mass'][idx]

        core_radius_1 = df_halo_parameter['core_radius_1'][idx]/current_time_a
        x = np.logspace(-1, 3, num=50)
        plt.plot( x, soliton(x) )


for idx in range(idx_start, idx_end+1, didx):

    current_time_z = df_Halo_Parameter['time'][idx]
    current_time_a = 1/(current_time_z+1)
    # core_radius_1 = df_Halo_Parameter['core_radius_1'][idx]/current_time_a

    
    # def r1(x):
    #     return x**-1*5e6

    # def r3(x):
    #     return x**-3*5e8

    # x = np.logspace(-1, 3, num=50)
    # anal = soliton(x)

    # ref1 = r1(x)
    # ref3 = r3(x)
    plot_profile(compare_path, 'gadget', False)
    plot_profile('./', 'higher resolution')
    # plot_profile('/projectV/vivi235711/gamer/L_2.8_N_256_seed_260757_check_dens/plot_script', 'gamer_hybrid')
    # plot_profile('/projectV/vivi235711/gamer/L_2.8_N_256_seed_260757/plot_script', 'gamer_hybrid')
    plot_profile('/projectZ/vivi235711/projectV_backup/gamer/m22_0.2_L_2.8_N_256_seed_705107_hybrid/plot_script', 'lower resolution')


    # plt.plot( radius, dens, '.', label='Data_%02d'%idx,color=colorVal)
    # plt.plot( radius, dens, 'bo', label='simulation')
    # plt.plot( x, anal, label='analytical' )
    # plt.plot( x, ref1, '--' , label='r^-1' )    
    # plt.plot( x, ref3, '--' , label='r^-3' )
    plt.plot( [1e-1, 1e3], [4096, 4096], '--')
    plt.plot( [1e-1, 1e3], [512, 512], '--')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-1,1e3)
    plt.ylim(1e0,1e9)
    plt.ylabel('$\\rho(r)/\\rho_{m0}$')
    plt.xlabel('radius(kpc/a)')
    plt.legend(loc = 'upper right')
    # plt.title('z = %.2e core radius = %.2f kpc/a'%(current_time_z, core_radius_1), fontsize=12) 
    # plt.title('density profile', fontsize=12) 


    FileOut = 'fig_profile_density_%d_%02d'%(halo, idx)+'.png'
    plt.savefig( FileOut)
    plt.close()
    
# FileOut = 'fig_profile_density'+'.png'
# plt.savefig( FileOut)
