# this is used to launch a flask web interface to control the beamscan and image generation process
# uses flask to create a web interface to control the process

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import signal
import time
import shutil

from experiment_manager import ExperimentSystemManager

# first create the flask app to avoid reloading the ExperimentSystemManager
app = Flask(__name__)
# app.secret_key = 'secret_key' # set the secret key for the session, needed for flash messages

# create an experiment system manager
print("Welcome!")
print("Creating the experiment system manager.")
try:
    esm = ExperimentSystemManager()
except Exception as e:
    print(f"Error creating the experiment system manager:\n{e}")
    exit(1)
# manually set the datapath for now TODO: make this updateable from the web interface
esm.datapath = "/home/sunlab/beamscan_data"
time.sleep(5) # wait for the GNU Radio process to start
if esm.gnu_service.poll() is not None: # make sure the GNU Radio process is running
    print("GNU Radio process is not running, exiting.")
    esm.shutdown()

#check if the directory exists
if not os.path.exists(esm.datapath):
    print(f"Directory {esm.datapath} does not exist.")
    os.makedirs(esm.datapath)
    print(f"Created directory {esm.datapath}")
else:
    print(f"Experiment Data Directory Found: {esm.datapath}")


def get_newest_image(suffix:str=".png",directory:str="") -> str:
    '''walk through the directory and find the newest image file with the given suffix'''
    image_path = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                if image_path is None:
                    image_path = os.path.join(root, file)
                else:
                    if os.path.getctime(os.path.join(root, file)) > os.path.getctime(image_path):
                        image_path = os.path.join(root, file)
    if image_path is None:
        print(f"No image with suffix {suffix} found in the directory {directory}")
        return None
    return image_path

# catch the ctrl+c signal and shutdown the experiment system manager before exiting
def signal_handler(sig, frame):
    print('Shutting down the experiment system manager.')
    esm.shutdown()
    print('Shutdown complete.')
    exit(0)
signal.signal(signal.SIGINT, signal_handler) 



@app.route('/')
def index():
    # get the newest images
    image_path_mean = get_newest_image(suffix="_gp_heatmap.png", directory=esm.datapath)
    image_path_mse = get_newest_image(suffix="_gp_heatmap_std.png", directory=esm.datapath)
    image_path_camera = get_newest_image(suffix="_camera.jpg" , directory=esm.datapath)
    print(f'{image_path_mse=}'); print(f'{image_path_mean=}'); print(f'{image_path_camera=}')
    # if found, copy and overwrite the images to the static directory
    if image_path_mean is not None:
        base_timestamp = os.path.basename(image_path_mean).split('_')[0] # this is the timestamp
        destination_path_mean = 'static/gp_heatmap.png'
        shutil.copy(image_path_mean, destination_path_mean)
    else:
        base_timestamp = 'No Scan Data'
        destination_path_mean = 'static/no_scan_placeholder.png'

    if image_path_mse is not None:
        destination_path_mse = 'static/gp_heatmap_std.png'
        shutil.copy(image_path_mse, destination_path_mse)
    else:
        destination_path_mse = 'static/no_scan_placeholder.png'

    if image_path_camera is not None:
        destination_path_camera = 'static/camera.jpg'
        shutil.copy(image_path_camera, destination_path_camera)
    else:
        destination_path_camera = 'static/no_scan_placeholder.png'
    
    return render_template('index.html', 
                           image1=destination_path_mean, 
                           image2=destination_path_mse,
                           image3=destination_path_camera,
                           base_time=base_timestamp,
                           base_datapath=esm.datapath)

@app.route('/beamscan', methods=['POST'])
def beamscan():
    '''Start the beamscan process. 
    Beamscan -> Save Beamscan Data -> Save Camera Image -> Fit Gaussian Process Model'''
    
    print("Beamscan started.",end=" ")
    esm.rx_beamscan()
    print("Beamscan finished.", end=" ")
    esm.save_beamscan_data()
    print("Beamscan data saved.", end=" ")
    esm.save_camera_image()
    print("Camera image saved.", end=" ")
    print("Beamscan process finished. Fitting Gaussian Process model.")
    esm.vis_gp_heatmap()
    print("Gaussian Process model fitted and saved.")
    # flash(f"Beamscan process completed.\n Saved data with base_filename {esm.base_filename}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # start the flask app, use_reloader=False to prevent the app from restarting on file changes
    app.run(debug=True, use_reloader=False, host='0.0.0.0') 