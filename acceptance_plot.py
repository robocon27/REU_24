import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('acceptance.txt') # Columns: Energy, acceptance, error
lgE = data[:,0]
acceptance = data[:,1]
error = data[:,2]

plt.errorbar(lgE,acceptance,yerr = error)
plt.yscale('log')
plt.xlabel('lg(Energy/GeV)',fontsize =16)
plt.ylabel('Acceptance($m^2$sr)',fontsize =16)
plt.savefig('acceptance_plot.pdf',bbox_inches='tight')

plt.clf()
plt.close()

data = np.loadtxt('event_rate.txt') # Columns: Energy, rate, error
lgE = data[:,0]
rate = data[:,1]
error = data[:,2]

plt.errorbar(lgE,rate,yerr = error)
plt.yscale('log')
plt.xlabel('lg(Energy/GeV)', fontsize = 16)
plt.ylabel('Event Rate(1/yr)',fontsize =16)
plt.title('Total Event Rate = %.1f CR per year' % rate.sum())
plt.savefig('event_rate_plot.pdf',bbox_inches='tight')