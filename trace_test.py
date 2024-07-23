import matplotlib.pyplot as plt
import numpy as np
import glob
import logging
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.utilities import units
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.framework.base_station

logging.basicConfig(level=logging.INFO)
file_pattern = '/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/cr_fullTrigger_lgE*/*.nur'
files = glob.glob(file_pattern)
#file = '/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/cr_lgE9.5_zenith5_0.nur'
nuradioreco_io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(files)


n_deep_ids = 15
n_events = nuradioreco_io.get_n_events()

event_num = -1
trace_list = []
Vrms = []
Vpp = []
SNR = []
SNR_trig=[]
bool_trig_out = []
for event in nuradioreco_io.get_events():
    #print(event)
    station = event.get_station(11)
    event_num += 1
    event_trace = []
    channel_SNR = []
    for id in station.get_channel_ids():
        if(id >= 12 and id <= 20):
            continue
        #if(id >= 4):
            #continue
        #print(id)
        channel = station.get_channel(id)
        t = channel.get_times()
        trace = channel.get_trace()
        trace_list.append(trace)
        event_trace.append(trace)
        Vmax = np.max(trace)
        Vmin = np.min(trace)
        dV = Vmax - Vmin
        rms = np.sqrt(np.mean(np.square(trace)))
        if id < 4:
            channel_SNR.append(dV/2/rms)

        lbl = str("Ch %d" % id)
        plt.plot(t, trace, label=lbl, rasterized=True)
        #break


    
    all_traces = np.concatenate(event_trace)
    Vpp.append(dV)
    rms = np.sqrt(np.mean(np.square(all_traces)))
    Vrms.append(rms)
    SNR_val = np.mean(channel_SNR)
    SNR.append(SNR_val)
    triggered = station.has_triggered('deep_high_low')
    #triggered = station.has_triggered('high_low')
    if triggered == True:
        SNR_trig.append(SNR_val)
    bool_trig_out.append(triggered)


        
    """
    plt.xticks(fontsize = 12.5)
    plt.yticks(fontsize = 12.5)
    plt.xlabel("Time(ns)",fontsize = 16)
    plt.ylabel("Voltage(V)",fontsize = 16)
    plt.legend(loc=0, ncols=4, fontsize='small')
    #plt.ylim(-1e-2, 1e-2)
    #plt.title("Voltage Trace vs Corresponding Time($10^{18.5}$ eV, 5$^{\circ}$ zenith and SNR = %.1f)" % SNR_val, fontsize = 14)
    plt.title("Voltage Trace vs Corresponding Time", fontsize = 19)
    outfile = str("trace_test_18.5eV_5zenith(1)/trace_test_ev_noise_%s.pdf" % event_num)
    plt.savefig(outfile,bbox_inches='tight')
    plt.clf()
    plt.close()
    
"""

for i, SNR_val in enumerate(SNR):
    print(f"Event {i}: SNR = {SNR_val}: Trigger = {bool_trig_out[i]}: Vpp = {Vpp[i]}")
#print(Vpp)
print("Max Peak to peak:",np.max(Vpp))
# Bin the SNR values

numeric_trig_out = [1 if x else 0 for x in bool_trig_out]
#print(numeric_trig_out)
nbins = int(np.max(SNR)/0.5+1)
event_counts, bin_edges = np.histogram(SNR, bins=nbins, range=(0, np.max(SNR)))
trigger_counts, bin_edges = np.histogram(SNR_trig, bins=nbins, range=(0, np.max(SNR)))
print(bin_edges)
binCenters = (bin_edges[1:] + bin_edges[:-1])/2.0
efficiency = trigger_counts/event_counts 
binCenters = binCenters[~np.isnan(efficiency)]
event_counts = event_counts[~np.isnan(efficiency)]
efficiency = efficiency[~np.isnan(efficiency)]
print(efficiency)
#Assuming poison distribution 
errors = 1/np.sqrt(event_counts)

plt.errorbar(binCenters,efficiency,yerr=errors)
plt.axhline(y=0, c='k', linestyle='--')
plt.axhline(y=1, c='k', linestyle='--')
plt.xlabel('Average SNR',fontsize =16)
plt.ylabel('Trigger Efficiency',fontsize =16)
plt.title("Trigger Efficiency vs Average SNR")
plt.savefig("Trigger_Efficiency.pdf")

#print(event_counts)
#print(trigger_counts)


