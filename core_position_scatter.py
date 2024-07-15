import sys
sys.path.insert(1,'/mnt/c/Users/moonm/REU 24/myenv/Lib/site-packages/RadioPropa')
import radiopropa
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
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
logger = logging.getLogger()
logger.setLevel(logging.INFO)
parser = argparse.ArgumentParser(description='Run air shower Reconstruction')
parser.add_argument('--detector_file', type=str, nargs='?', default='/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/RNO_season_2023.json',
                    help='choose detector with a single station for air shower simulation')
parser.add_argument('--input_file', type=str, nargs='?',
                    default='/mnt/c/Users/moonm/REU 24/myenv2/lib/python3.10/site-packages/NuRadioReco/examples/cr_data/SIM000074.hdf5', help='hdf5 coreas file')
args = parser.parse_args()

logger.info(f"Use {args.detector_file} on file {args.input_file}")
det = Detector(json_filename=args.detector_file,antenna_by_depth=False)

print(args.detector_file)
print(det)

station_id = det.get_station_ids()[0]
print(station_id)
t0 = datetime.datetime.now()
det.update(t0)

# module to read the CoREAS file and convert it to NuRadioReco event, each observer is a new event with a different core position
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([args.input_file], station_id, debug=False)

all_cores = []
#avgPos = [-1572.8161736544128,729.3722386485937,0]
#newPos = [0,0,0]

for evt in readCoREASStation.run(detector=det):
    sim_shower = evt.get_first_sim_shower()
    core_position = sim_shower[shp.core]
    all_cores.append(core_position)

all_cores = np.asarray(all_cores)
#print(all_cores)

average_x = np.mean(all_cores[:,0])
average_y = np.mean(all_cores[:,1])
print(average_x)
print(average_y)

avgPos = [average_x,average_y,0]
newPos = [0,0,0]

all_cores = []
for evt in readCoREASStation.run(detector=det):
    sim_shower = evt.get_first_sim_shower()
    core_position = sim_shower[shp.core]
    all_cores.append(core_position)
    core_position -= avgPos
    core_position += newPos

all_cores = np.asarray(all_cores)
#print(all_cores)

average_x = np.mean(all_cores[:,0])
average_y = np.mean(all_cores[:,1])


plt.scatter(all_cores[:,0], all_cores[:,1])
plt.scatter(average_x, average_y, color='red', label='Average Position', s=100)
plt.axhline(average_y, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(average_x, color='gray', linestyle='--', linewidth=0.7)

plt.savefig("core_pos_test_75deg.pdf")
