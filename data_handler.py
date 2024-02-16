import numpy as np
import pandas as pd
from records import TERMINAL_VELOCITIES, RADIUS


def scale_values(data, scale, frame_rate, time_scale):
    """
    This function scales the x, y, and time values in the given data. Loosely speaking, the scaled x and y should have unit in mm, and the scaled time should have unit in second.
    :param data: The original data
    :param scale: The 3-D scale factor for x and y
    :param frame_rate: Frame rate of the video, for converting frame number to time
    :param time_scale: The timescale tau (tau = r/v_light)
    :return: Scaled x, y, and time values
    """
    scaled_x = data['x'] / scale
    scaled_y = data['y'] / scale
    scaled_time = (data['frame'] / frame_rate) / time_scale

    return scaled_x, scaled_y, scaled_time


def normalize_time(frame_num, frame_rate, name_light):
    """
    This function normalizes the time series of the data by its defined timescale.
    :param frame_num: Total number of frames in the data
    :param frame_rate: Frame rate of the camera
    :param name_light: Name of the lightest sphere in the particle
    :return: Normalized time
    """
    v_light_term = TERMINAL_VELOCITIES[name_light]  # terminal velocity of the lightest particle
    tau = RADIUS / v_light_term  # timescale of the sedimentation process
    # normalize the times by the timescale tau
    normalized_time = np.arange(frame_num) / (frame_rate * tau)
    return normalized_time


def rolling_average(data, neigh_num):
    """
    Calculate the moving average of flame depth with pandas.rolling() method. Larger neighbor number makes the array shorter because the beginning and ending elements are NaNs.
    :param data: the original data array before smoothening.
    :param neigh_num: number of neighbors on each side to be averaged.
    :return: smoothened data array with the first and last neighbors being NaNs
    """
    # calculate the window size according to the neighbor number
    window_size = 2*neigh_num + 1

    # determine the averaged data array by using the pandas.rolling() method
    new_data = pd.Series(data).rolling(window_size, center=True).mean().values

    return new_data


def five_point_stencil(t, data):
    """
    Calculate the derivative using five-point stencil method
    :param t: original time array before differentiation
    :param data: original data array before differentiation
    :return: arrays of time and derivative without the first and the last several elements.
    """

    # since this method uses the surrounding four elements around the desired point,
    # the derivative array will be four elements shorter than the original array.
    new_t = t[2:-2]
    derivative = np.empty(len(data) - 4)

    h = t[1] - t[0]
    for i in range(2, len(data) - 2):
        derivative[i - 2] = (-data[i + 2] + 8 * data[i + 1] - 8 * data[i - 1] + data[i - 2]) / (12*h)

    return new_t, derivative


def forward_difference(t, data):
    """
    Calculate the forward discrete difference
    :param t: original time array before differentiation
    :param data: original data array before differentiation
    :return: arrays of time and derivative without the first element.
    """
    h = t[1] - t[0]  # the time spacing
    new_t = t[:-1]

    derivative = [(data[i+1] - data[i]) / h for i in range(len(data) - 1)]

    return new_t, np.array(derivative)