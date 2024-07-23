import matplotlib.pyplot as plt
import numpy as np
import glob
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.utilities import units
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.framework.base_station


def J(E):  # Auger spectrum from PoS(ICRC2021)324 equation 1

  # E must have units of eV

  J0 = 8.34e-11 # 1/km2/sr/yr/eV
  E0 = 1e16 # eV

  Eij = [2.8e16, 1.58e17, 5.0e18, 1.4e19, 4.7e19] # eV
  gammai = [3.09, 2.85, 3.283, 2.54, 3.03, 5.3]
  omegaij = [0.25, 0.25, 0.05, 0.05, 0.05]

  flux = J0 * (E/E0)**(-gammai[0])
  for i in range(5):
    flux *= (1.0 + (E/Eij[i])**(1/omegaij[i]) )**((gammai[i]-gammai[i+1])*omegaij[i])

  return flux # in units of 1/km2/sr/yr/eV



lgE = np.arange(7,10,0.5)
zenith = np.arange(5,90,5)
dzenith = 5
Aeff = np.zeros([len(lgE)*len(zenith),3])
acceptance = np.zeros([len(lgE),3])
Afid = 2*np.pi*250**2
idx = 0
rate = np.zeros([len(lgE),3])


for i in range(len(lgE)):
    for j in range(len(zenith)):
        file_pattern = str('/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/cr_fullTrigger_lgE%.1f/cr_fullTrigger_lgE%.1f_zenith%d_*.nur' % (lgE[i],lgE[i],zenith[j]))
        files = glob.glob(file_pattern)
        print(files)
        if len(files) == 0:
            Aeff[idx,0] = lgE[i]
            Aeff[idx,1] = zenith[j]
            Aeff[idx,2] = -1.0
            idx += 1
            continue

        nuradioreco_io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(files)
        n_triggered = 0.0
        for event in nuradioreco_io.get_events():
            station = event.get_station(11)
            triggered = station.has_triggered(f'deep_high_low')
            if triggered == True:
                n_triggered += 1.0
            #print(n_triggered)
        # rand offset = 5 and 240 observer positions
        n_thrown = 5*240*len(files)
        Aeff[idx,0] = lgE[i]
        Aeff[idx,1] = zenith[j]

        if n_thrown == 0:
            Aeff[idx,2] = -1.0
        else:
            Aeff[idx,2] = (n_triggered/n_thrown)*Afid
        
        if Aeff[idx,2] > 0:
            acceptance[i,1] += 2*np.pi*Aeff[idx,2]*(np.cos((zenith[j]-dzenith/2)*np.pi/180)-np.cos((zenith[j]+dzenith/2)*np.pi/180))
            Aeff_error = (np.sqrt(max(1,n_triggered))/n_thrown)*Afid
            acceptance[i,2] += (2*np.pi*Aeff_error*(np.cos((zenith[j]-dzenith/2)*np.pi/180)-np.cos((zenith[j]+dzenith/2)*np.pi/180)))**(2)
        
        idx += 1
    acceptance[i,0] = lgE[i]
    acceptance[i,2] = np.sqrt(acceptance[i,2]) 
    rate[i,0] = lgE[i]
    flux = J(10**(lgE[i]+9))
    flux *= 10**-6
    d_rate = flux*acceptance[i,1]
    de = 10**(lgE[i]+0.25+9) - 10**(lgE[i]-0.25+9)
    rate[i,1] = d_rate*de
    d_rate = flux*acceptance[i,2]
    rate[i,2] = d_rate*de

print(rate[:,1].sum())
np.savetxt('event_rate.txt',rate) #Columns: lgE(GeV), rate per year, error of rate per year
np.savetxt('eff_area_calc.txt',Aeff)
np.savetxt('acceptance.txt',acceptance)








