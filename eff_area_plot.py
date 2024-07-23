import matplotlib.pyplot as plt
import numpy as np


zenith = np.arange(5,90,5)
n = len(zenith)
colors = plt.cm.viridis(np.linspace(0,1,n))
data = np.loadtxt('eff_area_calc.txt') #Columns: energy, zenith, Aeff
for i in range(len(zenith)):
    mask = data[:,1] == zenith[i]
    lgE = data[mask,0] 
    Aeff = data[mask,2]
    lgE = lgE[Aeff > 0]
    Aeff = Aeff[Aeff > 0]
    #plt.figure(font_size=18)
    plt.plot(lgE,Aeff,label=str(r'$\theta = %d $' % zenith[i]),color = colors[i])


plt.yscale('log')
plt.yticks()
#plt.xlabel('lg(Energy/GeV)',fontsize = 18)
plt.xlabel('lg(Energy/GeV)',fontsize = 16)
plt.ylabel('Effective Area($m^2$)',fontsize =16)
plt.legend(loc=0)
plt.savefig('eff_area_plot.pdf')