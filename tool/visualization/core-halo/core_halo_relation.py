import yt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read data
df = pd.read_csv( 'Halo_Parameter' , sep = '\s+' , header = 0 )

# Mmin
Mmin_8 = 4.4*10**7*0.8**(-3/2)
Mmin_4 = 4.4*10**7*0.4**(-3/2)
Mmin_2 = 4.4*10**7*0.2**(-3/2)

FileOut = 'core_halo.png'

# analytical prediction
def line(x):
    return (1/4.0)*x**(1/3.0)
   
t = np.arange(1.0e0 , 6.0e5 , 1.0e3)
plt.plot( t , line(t)  , 'k--', lw=1 )

# plot data
d, = plt.plot( df['halo_chmr'] , df['core_chmr'] , 'r.' , ms=5)
# d2, = plt.plot( df_2['halo_chmr'] , df_2['core_chmr'] , marker = 'p' , mfc = '#ff7fff' , ls = 'None' , lw=1) 

# figure setting
plt.xscale( 'log', nonposx='clip' )
plt.yscale( 'log', nonposy='clip' )
plt.xlim( 1.0e0 , 1.0e4 )
plt.ylim( 1.0e-1, 1.0e1 )
plt.xlabel( r'$(\zeta({z})/\zeta({0}))^{1/2}\mathrm{M}_{h}/\mathrm{M}_{min}\:(\mathrm{M}_\odot)$', fontsize = 10)
plt.ylabel( r'$\mathrm{a}^{1/2}\mathrm{M}_{c}/\mathrm{M}_{min}\:(\mathrm{M}_\odot)$', fontsize = 10 )
plt.annotate( '$M_{min0.2}=$' + '%.3e'%Mmin_2 , xy = (0.03,0.95) , xycoords='axes fraction')
plt.annotate( '$M_{min0.4}=$' + '%.3e'%Mmin_4 , xy = (0.03,0.90) , xycoords='axes fraction')
plt.annotate( '$M_{min0.8}=$' + '%.3e'%Mmin_8 , xy = (0.03,0.85) , xycoords='axes fraction')
plt.legend([d2,d4,d8],['0.2','0.4','0.8'],loc = 'upper right')
# for i in range(0,len(df_8['time']),1):
#    plt.annotate('%.2f'%df_8['time'][i], (df_8['halo_chmr'][i], df_8['core_chmr'][i]), size = 4)
    
plt.savefig( FileOut, bbox_inches = 'tight', pad_inches = 0.05, dpi = 150)

