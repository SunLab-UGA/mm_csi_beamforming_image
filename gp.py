# this creates a Gaussian Process object to abstract the Gaussian Process regression model.

import pickle
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import glob
import os
import time
import numpy as np

from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process.kernels import Kernel

from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore") # ignore the warnings from the GP



class GaussianProcess():
    '''Gaussian Process class to abstract the fucltionality of the Gaussian Process regression model
    for the beamscan data into a single spatial spectrum image'''
    def convert_to_cartesian(self, theta, phi):
        '''convert the spherical coordinates to cartesian coordinates'''
        x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = np.cos(np.deg2rad(theta))
        return x, y, z
    def create_linespace(self, xx_range:int, yy_range:int, resolution:int=100):
        '''create a linespace for the x and y coordinates'''
        xx = np.linspace(xx_range[0], xx_range[1], resolution)
        yy = np.linspace(yy_range[0], yy_range[1], resolution)
        aa = np.array(list(product(xx, yy)))
        return xx,yy,aa
    
    def extract_plot_data(self, data: list[dict],csi_channel:str|int='avg') -> tuple[np.ndarray, np.ndarray]:
            '''Extract the plot data from the given data
            data: the loaded data from the pickle file
            csi_channel: the channel to extract the data from, default is 'avg'
                the magnitude is calculated regardless of the channel selection
            returns: the theta_phi and avg_csi_mag as numpy arrays'''
            # parse the csi_channel
            if csi_channel == 'avg':
                csi_channel = 'avg_csi'
            elif isinstance(csi_channel, int):
                if csi_channel >= 0 and csi_channel <= 52:
                    channel_index = csi_channel
                    csi_channel = f'csi'
            else:
                raise ValueError("csi_channel must be 'avg' or 'int (0-52)'")
            csi_min = float('inf'); csi_max = float('-inf') # initialize the min and max values
            theta_phi = []
            z_axis = []
            csi_mag = []
            for entry in data:
                if entry[csi_channel] is None: # check if the csi contains None (the packet was not received)
                    continue # skip the entry
                if csi_channel == 'avg_csi':
                    csi_mag.append(np.abs(entry['avg_csi']))
                    csi_max = max(csi_max, np.abs(entry['avg_csi'])) # track min/max
                    csi_min = min(csi_min, np.abs(entry['avg_csi']))
                else:
                    csi_mag.append(np.abs(entry['csi'][channel_index]))
                    csi_max = max(csi_max, np.abs(entry['csi'][channel_index]))
                    csi_min = min(csi_min, np.abs(entry['csi'][channel_index]))
                # Convert spherical coordinates to Cartesian coordinates for prediction and plotting
                x, y, z = self.convert_to_cartesian(entry['beam']['theta'], entry['beam']['phi'])
                theta_phi.append([x, y])
                z_axis.append(z) # the z-axis here is the curvature of the sphere (if needed)
            
            # convert the lists to numpy arrays
            theta_phi = np.array(theta_phi, dtype=np.float64)
            csi_mag = np.array(csi_mag, dtype=np.float64)

            if len(theta_phi) == 0:
                raise ValueError("No data to plot")
            else:
                print(f"Data Parsed, Num_Packets: {len(theta_phi)}")
                print(f"CSI Magnitude Range: {csi_min} - {csi_max}")
            return theta_phi, csi_mag
    
    def __init__(self, data: list[dict], linespace_density:int=180,
                 kernel:Kernel|None = None):
        '''initialize the Gaussian Process object
        WARNING: THIS HAS LOTS OF HARDCODED VALUES!'''
        # extract the data
        self.theta_phi, self.csi_mag = self.extract_plot_data(data)
        # create the Gaussian Process object
        if kernel is None:
            self.kernel = C(0.0224**2) * RBF(length_scale=0.179) + WhiteKernel(2.79e-05)
            print("Using Default Kernel")
        else:
            self.kernel = kernel
        self.gp = GaussianProcessRegressor( kernel=self.kernel,
                                    optimizer='fmin_l_bfgs_b',
                                    n_restarts_optimizer=30,
                                    copy_X_train=True,
                                    random_state=42)
        
        self.linespace_density = linespace_density

        # plotting variables
        self.plot_z_max:float = 0.127 # experimentally determined values
        self.plot_z_min:float = 0 # this "should" be the floor of the CSI data, but the gp will predict below this value
        # this is "OK" because we clamp the image values to the from the min and max values

    def fit(self):
        '''fit the Gaussian Process model to the data'''
        self.gp.fit(self.theta_phi, self.csi_mag)
        print(f"Model Fitted, Kernel: {self.gp.kernel_}")
        self.self_score = self.gp.score(self.theta_phi, self.csi_mag) # the score of the model on the training data
        print(f"Model Score: {self.self_score}")
        # create the meshgrid for the prediction
        
        xx, yy, aa = self.create_linespace(xx_range=(-0.7,0.7), # experimentally determined values
                                           yy_range=(-0.7,0.7), 
                                           resolution=self.linespace_density)
        self.xx = xx; self.yy = yy; self.aa = aa # save the meshgrid with the object

        yy_pred, yy_std = self.gp.predict(aa, return_std=True)
        self.yy_pred = yy_pred; self.yy_std = yy_std # save the prediction and std with the object
    
    def save_image(self, data:np.ndarray, filename:str, dpi:int=100):
        '''plot/save the an output image files as a heatmap png
        dpi=100, linespace_density=180 gives a 180x180 pixel image
        z_min and z_max are the min and max values for the colorbar
        the data should be directly from the prediction (this function reshapes the data)'''
        # scan the data for any values outside the plot_z_min and plot_z_max, give a warning if found
        if np.min(data) < self.plot_z_min:
            print(f"Warning: Predicted data contains values below the plot_z_min ({self.plot_z_min}).\
                  MIN: {np.min(data)}")
        if np.max(data) > self.plot_z_max:
            print(f"Warning: Predicted data contains values above the plot_z_max ({self.plot_z_max}).\
                  MAX: {np.max(data)}")

        # reshape the data
        data = data.reshape((self.linespace_density, self.linespace_density))
        fig:Figure; ax:Axes # type hinting
        fig, ax = plt.subplots(figsize=(self.linespace_density/100, self.linespace_density/100), dpi=dpi)
        ax.axis('off')
        # leave colorbar as automatic scaling
        # ax.imshow(data, cmap='viridis', origin='lower')
        # manually scale the colorbar
        ax.imshow(data, cmap='viridis', origin='lower', vmin=self.plot_z_min, vmax=self.plot_z_max)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(filename, dpi=dpi)
        print(f"plot saved to {filename}")

    def save_pickle(self, filename:str):
        '''save the Gaussian Process object to a pickle file'''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"GP object saved to {filename}")

    # --------------------------------- EXTRA VISUALIZATION FUNCTIONS -------------------------------
    
    def plot_scatter(self, title="Scatter Plot",
                    filename=None, show=False):
        '''plot data as a plotly scatter plot
        This does NOT use the plot_z_max and plot_z_min variables!'''
        fig = px.scatter(x=self.theta_phi[:,0], y=self.theta_phi[:,1],
                         color=self.csi_mag,
                         labels={'color':'Magnitude'})
        fig.update_layout(title=title,
                        autosize=True,
                        xaxis_title='X',
                        yaxis_title='Y',
                        xaxis=dict(scaleanchor="y", scaleratio=1), # make the x and y axis the same scale
                        yaxis=dict(scaleanchor="x", scaleratio=1))
        if filename is not None:
            fig.write_html(filename)
            print(f"plot saved to {filename}")
        if show:
            fig.show()
    
    def plot_heatmap_gp(self, title="GP Prediction of Beamscan",
                        filename=None, show=False):
        '''plot the heatmap of the GP prediction
        DATA IS AUTO-SCALED!'''
        yy_pred_reshaped = self.yy_pred.reshape((self.linespace_density, self.linespace_density))
        fig = px.imshow(yy_pred_reshaped, x=self.xx, y=self.yy, color_continuous_scale='Viridis', origin='lower', labels={'color':'Magnitude'})
        fig.update_layout(title=title,
                        autosize=True,
                        xaxis_title='X',
                        yaxis_title='Y')
        if filename is not None:
            fig.write_html(filename)
            print(f"plot saved to {filename}")
        if show:
            fig.show()
    
    def plot_heatmap_gp_std(self, title="GP Prediction of Beamscan error",
                        filename=None, show=False):
        '''plot the heatmap of the GP prediction
        DATA IS AUTO-SCALED!'''
        yy_std_reshaped = self.yy_std.reshape((self.linespace_density, self.linespace_density))
        fig = px.imshow(yy_std_reshaped, x=self.xx, y=self.yy, color_continuous_scale='Viridis', origin='lower', labels={'color':'Magnitude'})
        fig.update_layout(title=title,
                        autosize=True,
                        xaxis_title='X',
                        yaxis_title='Y')
        if filename is not None:
            fig.write_html(filename)
            print(f"plot saved to {filename}")
        if show:
            fig.show()
    


    