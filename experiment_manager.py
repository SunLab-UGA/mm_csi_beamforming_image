# this is a class that unifies the experiment system into one object, ExperimentSystemManager

import logging
import logging.config
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pickle
from itertools import product

import signal

# import warnings
# warnings.filterwarnings("ignore") # ignore warnings from the GP


from tymtek_wrapper import TMY_service, UDBox, BBox5G # TymTek wrapper
from gnu_manager import GNURadioManager # start and stop the GNU Radio process
from trans import transceiver # send and receive data from the GNU Radio process
from camera import Camera
from gp import GaussianProcess

logging.basicConfig(
level=logging.DEBUG, # change to INFO for runtime logging
format="%(asctime)s - %(name)s [%(levelname)s] %(message)s",
handlers=[logging.StreamHandler(sys.stdout),]
)

logger = logging.getLogger("Main")
logger.info("Python v%d.%d.%d (%s) on the %s platform" %(sys.version_info.major,
                                            sys.version_info.minor,
                                            sys.version_info.micro,
                                            platform.architecture()[0],
                                            platform.system()))
logger.debug("Current working directory: %s" %os.getcwd())

root_path = Path(__file__).absolute().parent
prefix = "TLKCore/lib/"
lib_path = os.path.join(root_path, prefix)
if os.path.exists(lib_path):
    sys.path.insert(0, os.path.abspath(lib_path))
    logging.info("Importing from path: %s" %lib_path)
else:
    print("Importing error: %s does not exist" %lib_path)
    exit(1)

class TMYLogFileHandler(logging.FileHandler):
    """Handle relative path to absolute path"""
    def __init__(self, fileName, mode):
        super(TMYLogFileHandler, self).__init__(os.path.join(root_path, fileName), mode)

# from tlkcore.TLKCoreService import TLKCoreService
# from tlkcore.TMYBeamConfig import TMYBeamConfig
from tlkcore.TMYPublic import DevInterface,RetCode,RFMode,UDState,UDMState,BeamType,UD_REF 
logging.info("Successfully Imported TLKCoreServices")

# ----------------------------------------------------------------------- END OF IMPORTS

class ExperimentSystemManager:
    '''Class to manage the entire setup,
    the tymtek, gnuradio, and camera systems
    additonally it will handle training a GP and storing all data'''
    def __init__(self):
        self.udbox:UDBox
        self.txbbox:BBox5G
        self.rxbbox:BBox5G
        self.tmy_service:TMY_service
        #--
        self.gnu_service:GNURadioManager
        self.transceiver:transceiver
        #--
        self.camera:Camera

        # experiment trial/run data
        self.base_filename:str = time.strftime("%Y%m%d-%H%M%S")
        self.csi_data:np.ndarray # the data from each beamscan run
        self.gp:GaussianProcess
        self.datapath:str = "experiment_name" # relative path to save the data
        self.full_filename:str|None = None # the full filename for each run

        # experiment stats
        self.scan_start_times = [] #len = number of scans performed + 1

        # startup the services
        self.startup_tymtek()
        self.startup_gnuRadio()
        self.startup_transceiver()
        self.startup_camera()

    def startup_camera(self):
        '''start the camera object'''
        self.camera = Camera()

    def startup_gnuRadio(self):
        '''start the GNU Radio process'''
        conda_env = "radio_base"
        # path = "/home/sunlab/radioconda/share/gnuradio/examples/ieee802_11"
        path = "ieee802_11"
        python_filename = "wifi_transceiver_nogui.py"
        gnu_service = GNURadioManager(conda_env=conda_env, path=path, python_filename=python_filename)
        gnu_service.start()
        self.gnu_service = gnu_service

    def startup_tymtek(self):
        '''start the TymTek service and get the devices
        the all the parameters are hardcoded for now'''
        # assignment to the experiment system manager is done at the end for readability
        # create a service object
        serials = ['UD-BD22470039-24','D2245E027-28','D2245E028-28'] #udbox, txbox, rxbox
        aakits = [None, 'TMYTEK_28ONE_4x4_C2245E029-28', 'TMYTEK_28ONE_4x4_C2245E030-28'] # antenna kit for device
        logger.info("Looking for Devices with serial numbers: %s" %serials)

        udbox:UDBox; txbbox:BBox5G; rxbbox:BBox5G # typehinting for the devices
        service = TMY_service(serial_numbers=serials)
        if len(service.devices) != len(serials): logger.error("Failed to find all devices");raise Exception("Failed to find all devices")
        # get the devices from the service, will be in the order of the serials
        udbox, txbbox, rxbbox = service.devices

        freq_list = [28_000_000, 25_548_000, 2_452_000, 50_000] # 
        if udbox.basic_setup(freq_list): logger.info("UDBox setup complete")
        else: logger.error("UDBox setup failed")
        # enable the UDBox channels
        udbox.set_channel_state(1, 1); udbox.set_channel_state(2, 1)

        # setup the BBoxOne5G devices
        if txbbox.basic_setup(28.0, RFMode.TX, aakits[1]): logger.info("TX BBoxOne5G setup complete")
        else: logger.error("TX BBoxOne5G setup failed")
        if rxbbox.basic_setup(28.0, RFMode.RX, aakits[2]): logger.info("RX BBoxOne5G setup complete")
        else: logger.error("RX BBoxOne5G setup failed")

        self.udbox = udbox ; self.txbbox = txbbox ; self.rxbbox = rxbbox
        self.tmy_service = service

        # finally, set the tx/rx beams to boresight
        self.rxbbox.boresight(); self.txbbox.boresight()

    def startup_transceiver(self):
        '''start the transceiver object'''
        trans = transceiver(tx_port=64001, rx_port=64000)
        self.transceiver = trans

    def rx_beamscan(self, packets_per_beam:int=2):
        '''perform a beamscan using the bbox devices and the GNU Radio process
        returns a list of dictionaries containing the beam data'''
        # update the experiment stats and base filename
        self.scan_start_times.append(time.time())
        self.base_filename = time.strftime("%Y%m%d-%H%M%S")
        self.full_filename = f"{self.datapath}/{self.base_filename}"

        csi_data = []
        theta_step = 5; phi_step = 20
        rx_scanner = self.rxbbox.scan_raster_generator(theta_step=theta_step, phi_step=phi_step)
        # check if the scanner is did not return None (indicating an error), primes the scanner(generator)
        if rx_scanner is None: logger.error("Failed to generate the scan raster");exit(1)

        # BEAMSCAN --------------------------------------------------------------------------
        logger.info("Performing beamscan")
        start_time = time.time()
        self.scan_start_times.append(start_time)
        for i, scanning in enumerate(rx_scanner):
            logger.debug(f'beam {i}: scanning {scanning}')
            if scanning is None: logger.info("scan ended");break # break if the scan is complete
            # transmit and receive a packet
            beam_data = {}
            for ii in range(packets_per_beam):
                pdu = f"HELLO SUNLAB {ii+1}!"
                beam_data['pdu'] = pdu
                beam_data['beam'] = self.rxbbox.beam # store the beam [gain, theta, phi]
                logger.debug(f"Transmitting packet: {pdu}")
                self.transceiver.send(pdu)
                beam_data['timestamp'] = time.time()
                time.sleep(0.001) # 1ms
                rx = self.transceiver.recieve_csi(timeout=7) # timeout in ms
                if rx is not None:
                    logger.debug("Received CSI data")
                    beam_data['csi'] = rx
                    # process the CSI data into an average
                    avg_csi = np.mean(rx, axis=0)
                    beam_data['avg_csi'] = avg_csi
                    logger.debug(f'avg_csi: {avg_csi}')
                else:
                    logger.debug("Failed to receive CSI data")
                    beam_data['csi'] = None
                    beam_data['avg_csi'] = None
                csi_data.append(beam_data)
                time.sleep(0.010) # 10ms
        end_time = time.time()
        beamscan_time = end_time - start_time
        logger.info(f"Time taken: {beamscan_time} seconds")
        self.csi_data = csi_data

    
    def vis_gp_heatmap(self):
        '''visualize the csi data as a heatmap using a gaussian process model
        and save the images to a file'''
        gp = GaussianProcess(self.csi_data)
        gp.fit()
        gp.save_image(data=gp.yy_pred, filename=f'{self.full_filename}_gp_heatmap.png')
        gp.save_image(data=gp.yy_std, filename=f'{self.full_filename}_gp_heatmap_std.png')
        gp.plot_scatter(filename=f'{self.full_filename}_scatter.html')
        gp.plot_heatmap_gp(filename=f'{self.full_filename}_gp_heatmap.html')
        gp.plot_heatmap_gp_std(filename=f'{self.full_filename}_gp_heatmap_std.html')
        gp.save_pickle(filename=f'{self.full_filename}_gp.pkl')

    def save_beamscan_data(self):
        '''save the latest beamscan data to a pickle file'''
        logger.info("len(csi_data): %d" %len(self.csi_data))
        logger.info("Saving the data to a pickle file")
        filename = f'{self.full_filename}_beamscan_csi.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.csi_data, file)
        logger.info(f"Data saved to {filename}")

    def save_camera_image(self):
        '''save an image from the camera'''
        self.camera.take_picture(filename=f'{self.full_filename}_camera.jpg')


    def shutdown(self):
        '''shutdown the system'''
        logger.info("Disabling UDBox channels")
        self.udbox.disable_channels()
        self.gnu_service.stop() # stop the GNU Radio process
        self.transceiver.close()
        self.camera.release()
        logger.info("ExperimentSystemManager Shutdown complete")


if __name__ == "__main__":
    print("=== Welcome! ===")
    print("testing the ExperimentSystemManager")

    esm = ExperimentSystemManager()
    esm.datapath = "/home/sunlab/beamscan_data"

    print("waiting for GNU Radio to start")
    time.sleep(5) # wait for the GNU Radio process to start
    esm.gnu_service.poll()

    #check if the directory exists
    if not os.path.exists(esm.datapath):
        print(f"Directory {esm.datapath} does not exist.")
        os.makedirs(esm.datapath)
        print(f"Created directory {esm.datapath}")
    
    # catch the ctrl+c signal and shutdown the experiment system manager before exiting
    def signal_handler(sig, frame):
        print('Shutting down the experiment system manager.')
        esm.shutdown()
        print('Shutdown complete.')
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # run the experiment -----------------------------------------------------------
    print("=== Starting the experiment ===")
    esm.rx_beamscan()
    print("=== Beamscan complete ===")
    esm.save_beamscan_data()
    esm.save_camera_image()
    print("=== Data saved ===")
    try:
        esm.vis_gp_heatmap()
    except Exception as e:
        print("Exception thrown visualizing the GP:\n", e)
    print("=== GP visualization complete ===")

    # run the experiment again -----------------------------------------------------------
    print("=== Starting the experiment again ===")
    esm.rx_beamscan()
    esm.save_beamscan_data()
    esm.save_camera_image()
    try:
        esm.vis_gp_heatmap()
    except Exception as e:
        print("Exception thrown visualizing the GP:\n", e)
    print("=== 2nd Experiment complete ===")
    # shutdown the system -----------------------------------------------------------
    print("=== Shutting down the experiment system manager ===")
    esm.shutdown()

    print("=== TEST Experiment completed! ;D ===")
