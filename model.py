''' Training a testing a neural network to predict peaks in a dataset. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tensorflow 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from display_data import load_data, clean, interpolate_nan
from sklearn.preprocessing import  MinMaxScaler 
from scipy.signal import resample

found_peak_filepaths = [
     "/Users/End User/Desktop/Kaito/Software/Numerical Analysis/Found Peaks/peak_indices_1.csv", 
     "/Users/End User/Desktop/Kaito/Software/Numerical Analysis/Found Peaks/peak_indices_2.csv"
 ]

raw_data_filepaths = [
    "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2025-01-31[11_37_44].csv",
    "/Users/End User/Desktop/Kaito/Data/Narrow_gap/2024-12-16[10_30_49].csv"
]
def plot_data(data, found_peaks, predicted_peaks):
    for i in range(len(data)):  # Adjust range for testing

        #Select row of data from each dataset
        y = data[i]
        found_peak_indices = found_peaks.iloc[i].values
        predicted_peak_indices = predicted_peaks[i]

         # Interpolate NaN values in the profile
        y = interpolate_nan(y) 
        x = np.arange(len(y))
        
        plt.figure(figsize=(18, 5))

        # Raw data with vertex predictions
        plt.xlim(0, len(x))
        #plt.ylim(-900, -450)
        plt.title(f"Raw Data {i+1} Data")
        plt.scatter(x, y, color="blue", s=3)
        # Plot the peaks on the raw data
        plt.scatter(found_peak_indices, y[found_peak_indices], color='red', s=50, marker='x', label='Found_Peaks')
        plt.scatter(predicted_peak_indices, y[predicted_peak_indices], color='yellow', s=50, marker='x', label='Predicted_Peaks')
        plt.legend()
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
 
def data_preprocessing(data):

    processed_rows = []

    for i in range (len(data)):
        row_data = data.iloc[i].values

        # Handle NaN values
        row_data = interpolate_nan(row_data)

        # Clean the data
        data_cleaned = clean(row_data, z_threshold=3)
        data_cleaned = interpolate_nan(data_cleaned)

        # Apply Gaussian smoothing
        data_smoothed = gaussian_filter1d(data_cleaned, sigma=8)     

        # Decide on a single, fixed length for all rows
        FIXED_LENGTH = 2048

        # Resample each row to the new fixed length
        resampled_row = resample(data_smoothed, FIXED_LENGTH)

        # Normalize each row to a [0, 1] range 
        min_val = np.min(resampled_row)
        max_val = np.max(resampled_row)
        
        # Avoid division by zero if a row is flat
        if (max_val - min_val) > 0:
            normalized_row = (resampled_row - min_val) / (max_val - min_val)
        else:
            # If the row is flat, it becomes all zeros
            normalized_row = resampled_row - min_val 


        processed_rows.append(normalized_row)

    return np.array(processed_rows)

def create_model(data):

    #Find number of features
    input_features = data.shape[1]

    #Set optimiser
    optimiser = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_features,)),
        Dropout(0.1), #Add dropout layer to prevent overfitting
        Dense(32, activation='relu'),
        Dense(3, activation='linear') #Predicting three peaks
    ])
    model.compile(optimizer=optimiser, loss='mse')
    return model

def combine_datasets(raw_data_list, peak_data_list):
    """
    Takes lists of dataframes/arrays, appends peaks to each raw data row,
    concatenates all datasets, and returns the final combined DataFrame.
    """
    if len(raw_data_list) != len(peak_data_list):
        print("Error: The number of raw datasets must match the number of peak datasets.")
        return None

    all_combined_data = []

    # Process each pair of raw data and peak dataframes/arrays
    for i in range(len(raw_data_list)):
        # Convert inputs to pandas DataFrames to ensure compatibility
        df_raw = pd.DataFrame(raw_data_list[i])
        df_peaks = pd.DataFrame(peak_data_list[i])

        print(f"Processing dataset {i+1}...")
        
        try:
            # Ensure the number of rows matches
            if len(df_raw) != len(df_peaks):
                print(f"  Warning: Row count mismatch in dataset {i+1}. Skipping this pair.")
                print(f"    Raw data has {len(df_raw)} rows.")
                print(f"    Peak data has {len(df_peaks)} rows.")
                continue

            # Drop the 'profile_index' column from the peaks data if it exists
            if 'profile_index' in df_peaks.columns:
                df_peaks = df_peaks.drop('profile_index', axis=1)
            
            # Reset index to ensure clean side-by-side concatenation
            df_raw.reset_index(drop=True, inplace=True)
            df_peaks.reset_index(drop=True, inplace=True)

            # Append the peak columns (y-values) to the raw data (X-values)
            df_combined = pd.concat([df_raw, df_peaks], axis=1)
            
            all_combined_data.append(df_combined)
            print(f"  Successfully combined raw data and peaks for dataset {i+1}.")

        except Exception as e:
            print(f"  An unexpected error occurred with dataset {i+1}: {e}")
            continue

    # Concatenate all processed datasets into one large dataframe
    if all_combined_data:
        print("\nConcatenating all processed datasets...")
        final_dataset = pd.concat(all_combined_data, ignore_index=True)
        
        print("-" * 30)
        print("✅ Data combination successful!")
        print("-" * 30)
        return final_dataset
    else:
        print("\nNo data was processed. Please check for error messages.")
        return None
    
def shuffle_and_split_data(combined_df):
    """
    Shuffles a DataFrame, splits it in half for training and testing,
    and separates features (X) from targets (y).
    """
    if combined_df is None:
        print("Input DataFrame is empty. Cannot split.")
        return None, None, None, None

    # 1. Shuffle the rows of the entire dataset randomly
    #    frac=1 means sample 100% of the data.
    #    reset_index(drop=True) cleans up the index after shuffling.
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nShuffled {len(shuffled_df)} total rows.")

    # 2. Split the shuffled DataFrame into two halves
    split_point = len(shuffled_df) // 2
    train_df = shuffled_df.iloc[:split_point]
    test_df = shuffled_df.iloc[split_point:]
    print(f"Split into {len(train_df)} training rows and {len(test_df)} testing rows.")

    # 3. Separate features (X) from targets (y)
    #    The last 3 columns are the peak locations (targets).
    X_train = train_df.iloc[:, :-3].values
    y_train = train_df.iloc[:, -3:].values

    X_test = test_df.iloc[:, :-3].values
    y_test = test_df.iloc[:, -3:] 

    print("Separated data into X (features) and y (targets).")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    #Define profile length
    PROFILE_LENGTH = 2048

    # Load and preprocess the data
    raw_data_1 = load_data(raw_data_filepaths[0], flip=False)
    raw_data_2 = load_data(raw_data_filepaths[1], flip=False)
    processed_1 = data_preprocessing(raw_data_1)
    processed_2 = data_preprocessing(raw_data_2)

    #Make a list of processed data
    processed_data = [processed_1, processed_2]

    # Load the found training and testing peaks
    found_peaks_1 = pd.read_csv(found_peak_filepaths[0])
    found_peaks_2 = pd.read_csv(found_peak_filepaths[1])
    found_peaks_1 = found_peaks_1[['peak_1', 'peak_2', 'peak_3']]
    found_peaks_2 = found_peaks_2[['peak_1', 'peak_2', 'peak_3']]

    # Make a list of found peaks
    found_peaks = [found_peaks_1, found_peaks_2]

    # Combine datasets
    data_combined = combine_datasets(processed_data, found_peaks)

    # Shuffle and split the combined dataset
    X_train, X_test, y_train, y_test = shuffle_and_split_data(data_combined)

    y_train = y_train / PROFILE_LENGTH

    # Create and train the model
    model = create_model(X_train)
    model.fit(X_train, y_train, epochs=19, validation_split=0.2)
    scaled_predicted_peaks = model.predict(X_test)
    predicted_peaks = (scaled_predicted_peaks * PROFILE_LENGTH).astype(int)
    predicted_peaks_sorted = np.sort(predicted_peaks, axis=1)

    print(predicted_peaks_sorted[0:10])
    print(y_test[0:10])

    # Plot the data and peaks
    plot_data(X_test, y_test, predicted_peaks_sorted)