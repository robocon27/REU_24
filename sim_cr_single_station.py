import sys
sys.path.insert(1,'/mnt/c/Users/moonm/REU 24/myenv/Lib/site-packages/RadioPropa')
import radiopropa
import math
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import h5py

import numpy as np
#import helper_cr_eff as hcr
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.io.coreas.readCoREASStation


import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
import matplotlib.pyplot as plt

import datetime
import logging
import argparse
logging.basicConfig()


logger = logging.getLogger('sim_cr_single_station')
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('--detector_file', type=str, nargs='?', default='/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/RNO_season_2023.json',
                    help='choose detector with a single station for air shower simulation')
parser.add_argument('--input_file', type=str, nargs='?',
                    default='/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/SIM000425.hdf5', help='hdf5 coreas file')

args = parser.parse_args()

logger.info(f"Use {args.detector_file} on file {args.input_file}")

det = Detector(json_filename=args.detector_file,antenna_by_depth=False)
t0 = datetime.datetime.now()
det.update(t0)

print(args.detector_file)
print(det)
station_id = det.get_station_ids()[0]
print(station_id)

shower = int(args.input_file.split("SIM")[1].split(".")[0])
shower = shower%5
data = h5py.File(args.input_file, "r")
print(data['inputs'].attrs['ERANGE'])
energy = data['inputs'].attrs['ERANGE'][0]
lgE = np.log10(energy)
zenith = data["inputs"].attrs["THETAP"][0]

print(shower,lgE,zenith)

# module to read the CoREAS file and convert it to NuRadioReco event, each observer is a new event with a different core position
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([args.input_file], station_id, debug=False)
print(readCoREASStation)
# module to set the event type, if cosmic ray, the refraction of the emission on the air ice interface is taken into account
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

# module to convolves the electric field with the antenna response
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

# module to add the detector response, e.g. amplifier, filter, etc.
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

# module to add thermal noise to the channels
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

# module to add galactic noise to the channels
# Galactic Noise not needed above 100MHz
#channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
#channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(0.01, 0.81, 0.1))

# module to simulate the trigger
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin()

# module to filter the channels
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

# module adjust the sampling rate of the electric field
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()

# module adjust the sampling rate of the channels
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

# module to write the event to a .nur file
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
outfile = str("cr_lgE%.1f_zenith%d_%d.nur" % (lgE,zenith,shower))
eventWriter.begin(outfile, max_file_size=2048)


#ray tracing properties
ref_index_model='tri_model'
icemodel = medium.get_ice_model(ref_index_model)
solution_types = propagation.solution_types
ray_tracing_module = propagation.available_modules[1]
prop = propagation.get_propagation_module(ray_tracing_module)
attenuation_model='GL1'
use_channels=[0,1,2,3]
use_MC=True
n_samples_multiplication_factor = 1


all_cores = []
for evt in readCoREASStation.run(detector=det):
    sim_shower = evt.get_first_sim_shower()
    core_position = sim_shower[shp.core]
    all_cores.append(core_position)

all_cores = np.asarray(all_cores)


average_x = np.mean(all_cores[:,0])
average_y = np.mean(all_cores[:,1])
#print(average_x)
#print(average_y)

avgPos = [average_x,average_y,0]
newPos = [0,0,-1e-9]
def random_point():
    while True:
        x = np.random.uniform(-500, 500)
        y = np.random.uniform(-500, 500)
        if math.sqrt(x**2 + y**2) < 500:
            return x, y

all_cores = []
#evt is CoREAS observer, station is RNO-G station
for evt in readCoREASStation.run(detector=det):
    #print("event", evt)
    for i in np.arange(5):
        newPos = [0,0,-1e-9]
        rand_x,rand_y = random_point()
        newPos = [rand_x,rand_y,-1e-9]

        for station in evt.get_stations():
            
            print("station", evt.get_stations())
            station.set_station_time(datetime.datetime.now())
            
            

            #eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')

            # Script for raytracing implented here before efieldToVoltageConverter for time delay
            # addTraceStartTime from framework->base_trace.py

            use_channels.sort()
            station_id = station.get_id()
            #print('station_id',station_id)
            noise_RMS = det.get_noise_RMS(station_id, 0) #assume noise is the same in all channels

            # n_expand will multiply the time window of the reconstructed trace by n.
            # This allows for this module to fit on any of the ray tracing solutions between the reconstructed traces and the true traces.
            # This is especially important for Moore's Bay where the ray tracing solutions can be microseconds apart. Setting n_expand to 50 for Moore's Bay suffices.
            n_expand = n_samples_multiplication_factor

            # assume all channles have same number of samples and sampling rate
            #print(station.get_channel_ids())

            '''
            Used for attenuation

            first_channel = station.get_channel(use_channels[0])
            n_samples = first_channel.get_number_of_samples() * n_expand
            sampling_rate = first_channel.get_sampling_rate()

            dt = 1./sampling_rate
            T = n_samples * dt  # trace length
            ff = np.fft.rfftfreq(n_samples, dt)

            '''
            
            
            if use_MC and (station.get_sim_station() is not None):
                
                # Get the ray tracing ids for simulated values
                # Note they are not ordered in any particular way
                sim_shower = evt.get_first_sim_shower()
                channels_with_existing_sol = set()
                for i_efield, efield in enumerate(station.get_sim_station().get_electric_fields()):
                    #print("here")
                    #print(efield.get_channel_ids())
                    #if efield.get_channel_ids()[0] in use_channels:
                    for id in efield.get_channel_ids():
                        channels_with_existing_sol.add(id)

                # Sometimes some events do not have ray tracing solutions to all channels that is requested with use_channels.
                # This block of code catches that so the following does not try to access data for channels which does not exist
                use_channels_tmp = []
                for channel in channels_with_existing_sol:
                    use_channels_tmp.append(channel)
                use_channels = use_channels_tmp
                use_channels.sort()
                #print(use_channels)

                shower_energy_sim = sim_shower[shp.energy] #inelasticity * nu_energy
                core_position = sim_shower[shp.core]
                all_cores.append(core_position)
                core_position -= avgPos
                core_position += newPos

            n_antennas = len(use_channels)
            antenna_positions = {}
            
            for iA, iCh in enumerate(use_channels):
                antenna_positions[iCh] = det.get_relative_position(station_id,iCh)
                #print(antenna_positions[iCh],iCh)
                
                

            # Used to initial numpy array data structures.
            maxNumRayTracingSolPerChan = 2
            n_reflections = 0
            if(attenuation_model == "MB1"):
                maxNumRayTracingSolPerChan = 6
                n_reflections = 1

            n_ray_tracing_solutions = np.zeros(n_antennas, dtype=int)
            launch_vectors = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 3))
            receive_vectors = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 3))
            travel_time = np.zeros((n_antennas, maxNumRayTracingSolPerChan))
            travel_distance = np.zeros((n_antennas, maxNumRayTracingSolPerChan))
            #attenuation = np.zeros((n_antennas, maxNumRayTracingSolPerChan, len(ff)))
            focusing = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 1))
            travel_time_min = np.zeros(n_antennas) + np.inf
            trig_channels = []
            for iA in antenna_positions:
                position = antenna_positions[iA]
                #print(iA, core_position, position)
                r = prop(icemodel, attenuation_model, n_frequencies_integration=25, n_reflections=n_reflections)
                r.set_start_and_end_point(core_position * units.m, position * units.m)
                r.find_solutions()
                print(r.get_number_of_solutions())
                if r.get_number_of_solutions() > 0 and iA in (0,1,2,3):
                    trig_channels.append(iA)

                n_ray_tracing_solutions[iA] = min(r.get_number_of_solutions(),maxNumRayTracingSolPerChan)
                for iS in range(r.get_number_of_solutions()):
                    launch_vectors[iA, iS] = r.get_launch_vector(iS)
                    receive_vectors[iA, iS] = r.get_receive_vector(iS)
                    travel_time[iA, iS] = r.get_travel_time(iS)
                    print(iA,iS,travel_time[iA, iS])
                    travel_time_min[iA] = min(travel_time_min[iA],r.get_travel_time(iS))
                    travel_distance[iA, iS] = r.get_path_length(iS)
                    #attenuation[iA, iS] = r.get_attenuation(iS, ff)
                    #focusing[iA, iS] = r.get_focusing(iS, 1*units.cm)
            
            #print(travel_time_min)
            initial_delay = np.zeros(len(travel_time_min))
            for iCh in range(len(travel_time_min)):
                if(np.isinf(travel_time_min[iCh])):
                        continue
                initial_delay[iCh] = det.get_cable_delay(station_id,iCh)
                print("Channel", iCh)
                print("Initial cable delay", initial_delay[iCh])

            delta_t = np.zeros(len(travel_time_min))
            for iCh in range(len(travel_time_min)):
                if(np.isinf(travel_time_min[iCh])):
                        continue
                print("channel", iCh)
                delta_t[iCh] = travel_time_min[iCh] - travel_time_min.min()
                delta_t[iCh] += det.get_cable_delay(station_id,iCh)
                det.set_cable_delay(delta_t[iCh],station_id,iCh)
                print("delay", delta_t[iCh])



            efieldToVoltageConverter.run(evt, station, det)

            channelGenericNoiseAdder.run(evt, station, det, amplitude=(1.091242302378349e-05)/4, min_freq=80*units.MHz, max_freq=800*units.MHz, type='rayleigh')
            #print(det.get_noise_RMS(station_id, 0))
            
            #channelGalacticNoiseAdder.run(evt, sta, det)

            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
            print("trig channels:",trig_channels)

            set_no_trigger = False
            if len(trig_channels) == 0:
                trig_channels = None
                set_no_trigger = True
            

            triggerSimulator.run(evt, station, det,
                                    threshold_high=5e-6,
                                    threshold_low=-5e-6,
                                    coinc_window=60,
                                    number_concidences=2,
                                    triggered_channels=trig_channels,
                                    trigger_name='high_low',
                                    set_not_triggered=set_no_trigger)
            
            print("Trigger outcome:", station.has_triggered('high_low'))

            channelResampler.run(evt, station, det, sampling_rate=3.2)

            electricFieldResampler.run(evt, station, det, sampling_rate=3.2)
            print("debug")

            for iCh in range(len(travel_time_min)):
                if(np.isinf(travel_time_min[iCh])):
                        continue
                det.set_cable_delay(initial_delay[iCh],station_id,iCh)
            
                
        eventWriter.run(evt, det=det, mode={
                    'Channels': True,
                    'ElectricFields': True,
                    'SimChannels': True,
                    'SimElectricFields': True
                })
        #break
#print(launch_vectors)
nevents = eventWriter.end()