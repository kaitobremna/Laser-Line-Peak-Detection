# Laser-Line-Peak-Detection
Attempts using both numerical and ML methods to detect seam edges from a laser line data points

Display data script uses numerical methods including a Savitzky–Golay filter to detect vertices using the second derivative once the data has been cleaned and smoothed. This provides predictions to a high accuracy.

Trained a NN using Tensorflow library and the predictions from display data, to decent accuracy, using a validation split to prevent overfitting. 
