''' Program to Display Data for testing purposes'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


''' -------
Load Data
---------- ''' 

filepaths = [
     "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2025-01-31[11_37_44].csv", 
     "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2024-12-16[10_30_49].csv",
    # "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2025-01-29[16_07_01].csv", 
    # "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2025-01-30[15_00_22].csv",
    #"/Users/joshkaura/Desktop/OSL/ML Analysis/RR Narrow Gap/Data/2025-02-05[14_13_17].csv",
    #"/Users/joshkaura/Desktop/OSL/ML Analysis/RR Narrow Gap/Data/2025-04-02[14_30_14].csv",
    #"/Users/End User/Desktop/Kaito/Data/VGroove/2021-02-10[07_57_42].csv", 
 ]

def load_data(filepath, flip=True):
    data = pd.read_csv(filepath)
    data = data.replace(0, np.nan)
    if flip:
        data = -data
    return data

#Function to interpolate NaN values in a profile
def interpolate_nan(profile):
    nans = np.isnan(profile)
    not_nans = ~nans
    if np.any(not_nans):
        indices = np.arange(len(profile))
        profile[nans] = np.interp(indices[nans], indices[not_nans], profile[not_nans])
    return profile

''' -----------
Insert Functions for Data Science Testing
-----------------'''
#Clean data
def clean(data, z_threshold=3):

    #Find zscore using median and MAD (input data has NaNs)
    y_median = np.nanmedian(data)
    
    #Find MAD
    absolute_deviations = np.abs(data - y_median)
    mad = np.median(absolute_deviations)

    #Calculate z-score
    z_score = 0.6745*(data - y_median) / mad if mad else 0

    # Identify outliers
    mask_outliers = np.abs(z_score) >= z_threshold

    # Remove outliers, replace as NaN
    data_cleaned = data.copy()
    data_cleaned[mask_outliers] = np.nan

    return data_cleaned

#Using a savgol filter to analyse data
def savgol_data(y_smoothed, deriv_window_length=350, deriv_polyorder=2):
    x_spacing = 1.0  # Example: Spacing between X values, in the same units as X data.

    #Calculate First Derivative 
    first_derivative = savgol_filter(
        y_smoothed,
        window_length=deriv_window_length,
        polyorder=deriv_polyorder,
        deriv=1,        
        delta=x_spacing # Scale correctly by X units
    )

    #Calculate Second Derivative (Curvature)
    second_derivative = savgol_filter(
        y_smoothed,
        window_length=deriv_window_length,  
        polyorder=deriv_polyorder,
        deriv=2,         
        delta=x_spacing  # Scale correctly by X units
    )

    return first_derivative, second_derivative

#Find peaks
def peaks(second_derivative_data, prominence_threshold=0.02, min_peak_distance=200):
    peaks_indices, properties = find_peaks(np.abs(second_derivative_data),
                                       prominence=prominence_threshold,
                                       distance=min_peak_distance # Add distance constraint
                                      )
    return peaks_indices, properties



''' ---------
Display Data
------------- '''

def plot_data(data):
    all_peaks = []
    for i in range(len(data)): # Adjust range for testing
        y = data.iloc[i].values

        # Interpolate NaN values
        y = interpolate_nan(y)  
 
        x = np.arange(len(y))
        
        plt.figure(figsize=(18, 5))

        y_smoothed = clean(y, z_threshold=3)

        # Interpolate NaN values after cleaning
        y_smoothed = interpolate_nan(y_smoothed)                                                  

        # Apply Gaussian smoothing
        y_smoothed = gaussian_filter1d(y_smoothed, sigma=8)

        # Apply Savitzky-Golay filter for analysis
        first_derivatives, second_derivatives = savgol_data(y_smoothed)

        # Find peaks in the second derivative
        peaks_indices, properties = peaks(second_derivatives, prominence_threshold=0.001, min_peak_distance=50)

        # Create a list with the found peaks, padded with NaN up to 3 elements
        peaks_to_save = list(peaks_indices[:3]) # Take the first 3 peaks if they exist
        while len(peaks_to_save) < 3:
             peaks_to_save.append(np.nan)

        # Save the peaks indices to a file
        all_peaks.append({
            'profile_index': i,
            'peak_1': peaks_to_save[0],
            'peak_2': peaks_to_save[1],
            'peak_3': peaks_to_save[2]
        })

        # Raw Data with vertex predictions
        plt.subplot(1, 3, 1)
        plt.xlim(0, len(x))
        #plt.ylim(-900, -450)
        plt.title(f"Raw Data {i+1} Data")
        plt.scatter(x, y, color="blue", s=3)
        # Plot the peaks on the cleaned data
        plt.scatter(peaks_indices, y_smoothed[peaks_indices], color='red', s=50, marker='x', label='Peaks')

        # Cleaned
        plt.subplot(1, 3, 2)
        plt.xlim(0, len(x))
        #plt.ylim(-900, -450)
        plt.title("Cleaned Data")
        plt.scatter(x, y_smoothed, color="blue", s=3)

        # Absolute Second derivative using savgol filter  
        plt.subplot(1, 3, 3)
        plt.xlim(0, 2000)
        #plt.ylim(-900, -450)
        plt.title(f"Second derivative")
        plt.scatter(x, np.abs(second_derivatives), color="blue", s=3)
        # Plot the peaks on the second derivative
        plt.scatter(peaks_indices, np.abs(second_derivatives[peaks_indices]), color='red', s=50, marker='x', label='Peaks')


        # Plot the line between two points
        plt.legend()
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close()

    return all_peaks


if __name__ == '__main__':
    for i in range(len(filepaths)):

        # Define the file path in a variable first
        output_path = fr'C:\Users\End User\Desktop\Kaito\Software\Numerical Analysis\Found Peaks\peak_indices_{i+1}.csv'

        filepath = filepaths[i]
        data = load_data(filepath, flip = False)

        found_peaks = plot_data(data)

        # Save the peaks to a CSV file
        peaks_df = pd.DataFrame(found_peaks)  #Saves into a DataFrame format from dictionary format
        peaks_df.to_csv(output_path, index=False)
        print(f"Saved peaks {i+1} to", output_path)