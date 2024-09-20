## Beamforming Imager

this project is to use two TYMTEK bbox5G and USRP SDR to collect 28GHz beamscan data along with a picture from the recievers POV.

TODO:
* create a beamscanning and visualization class
* integrate beamscanning and beamscanning visualization into one pipeline flow
* integrate image capture as well
* create a simple working capture frontend to aid in data collection

Tips:

* port stuck open?
    ```
    lsof -i :64001
    kill -9 [PID]
    ```
