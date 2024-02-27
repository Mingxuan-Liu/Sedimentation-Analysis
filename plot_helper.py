import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from skimage.transform import resize
from data_handler import *
from particle_helper import find_centroid
import json  # import the library that stores the particle configurations

with open('particle_configurations.json') as f:
    configurations = json.load(f)


def plot_particle(particle, name, ax, transparency=False):
    """
    Visualize a given particle in a 3D interactive domain.

    This function plots the particle's spheres, center of mass, and geometric center.
    It also creates a legend for the different materials used in the spheres.
    The function adjusts the limits of the plot according to the extents of the spheres.
    If `transparency` is set to True, the spheres will be semi-transparent.

    Parameters
    ----------
    particle : Particle
        The particle to visualize. It should have properties spheres (list of spheres
        with each sphere having center, radius, color, and material), center_of_mass (list of
        coordinates), center_of_geometry (list of coordinates), offset, and phi.
    name : str
        The name of the particle, which is used in the title of the plot.
    ax : matplotlib.axes.Axes
        The axes on which to plot the particle.
    transparency : bool, optional
        If set to True, the spheres will be plotted with a transparency of 0.5.
        If False (the default), the spheres will be fully opaque.

    Returns
    -------
    None

    """
    min_val = np.inf
    max_val = -np.inf
    # Create a list to hold the legend entries
    legend_elements = []

    for s in particle.spheres:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = s.center[0] + s.radius * np.outer(np.cos(u), np.sin(v))
        y = s.center[1] + s.radius * np.outer(np.sin(u), np.sin(v))
        z = s.center[2] + s.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        min_val = min(min_val, np.min(x), np.min(y), np.min(z))
        max_val = max(max_val, np.max(x), np.max(y), np.max(z))

        ax.plot_surface(x, y, z, color=s.color, alpha=0.5 if transparency else 1)

        # create a patch for the material if it's not already in the legend
        if s.material.capitalize() not in [element.get_label() for element in legend_elements]:
            legend_elements.append(Patch(facecolor=s.color, edgecolor=s.color, label=s.material.capitalize()))

    # Plot center of mass
    com = particle.center_of_mass
    mass_center = ax.scatter(com[0], com[1], com[2], color="r", s=30, marker='*')

    # Plot center of geometry
    cog = particle.center_of_geometry
    geometry_center = ax.scatter(cog[0], cog[1], cog[2], color="k", s=30, marker='x')

    # Create legend
    ax.legend([mass_center, geometry_center], ['Center of Mass', 'Geometric Center'])

    # Set title and capitalize each word as well as the element name
    title = name.title()
    ax.set_title(title + f", $\\chi={particle.offset:.2f}$")

    ax.set_xlim3d([min_val, max_val])
    ax.set_ylim3d([min_val, max_val])
    ax.set_zlim3d([min_val, max_val])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # add the legend
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(1.1, -0.15))

    ax.set_box_aspect([1, 1, 1])


def plot_principal_axes(com, eigenvectors, length, ax):
    """
    Plot the principal axes of the particle from its center of mass in 3D space.

    Parameters
    ----------
    com : numpy.ndarray
        The center of mass of the particle, given as a 3-element array.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot where the axes will be plotted.
    eigenvectors : numpy.ndarray
        The principal axes of the particle, given as a 3x3 array where each column is an eigenvector.
    length : float, optional
        The length of the vectors representing the axes in the plot.

    Returns
    -------
    None
    """
    # Define the colors and labels for the axes
    colors = ['r', 'g', 'b']

    # Create vectors for the principal axes, starting at the center of mass and
    # extending along each axis. Multiply by the length for visibility.
    for i in range(3):
        ax.quiver(com[0], com[1], com[2],
                  eigenvectors[0, i]*length, eigenvectors[1, i]*length, eigenvectors[2, i]*length,
                  color=colors[i], alpha=0.6, linewidth=2)


def plot_shadow(particle, image_size, center, scale=7):
    """
    Visualize the shadow of the particle on both 'xz' and 'yz' planes.

    Parameters:
    - particle: The ParticleSimulation object.
    - image_size: A tuple with the height and width of the experimental image in pixels.
    - center: A tuple with the x and y coordinates of the particle center in mm.
    - scale: The scale factor converting mm to pixels (default is 7).
    """
    # Generate the shadow grids
    shadow_grid_xz = particle.shadow('xz', image_size, center, scale)
    shadow_grid_yz = particle.shadow('yz', image_size, center, scale)

    # Create a plot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the shadow grid for 'xz' plane
    axs[0].imshow(shadow_grid_xz, cmap='gray')
    axs[0].axis('on')  # Changed to 'on' to see the axis if needed
    axs[0].set_title('Shadow on X-Z Plane')

    # Display the shadow grid for 'yz' plane
    axs[1].imshow(shadow_grid_yz, cmap='gray')
    axs[1].axis('on')  # Changed to 'on' to see the axis if needed
    axs[1].set_title('Shadow on Y-Z Plane')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_grayscale(frame):
    """
    This function plots the provided grayscale frame and its intensity histogram. It helps streamline the plotting
    process of grayscale images to manually check over and over again.
    :param frame: One frame of the grayscale images.
    :return: A 1 by 2 panel, with the left figure showing the grayscale image and right one showing its histogram.
    """
    # Set up the figure and axes for a side-by-side plot: one for the image, one for the histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image interactively
    ax1.imshow(frame, cmap='gray')
    ax1.set_title('Grayscale Image')

    # Calculate and display the histogram
    histogram, bin_edges = np.histogram(frame.flatten(), bins=256, range=[0, 256])

    # Plot the histogram
    ax2.plot(bin_edges[0:-1], histogram)
    ax2.set_title('Grayscale Histogram')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Count')

    plt.show()


def plot_motion(data, scale, frame_rate, time_scale, diff_method, avg_size, ax):
    """
    This function plots the motion properties of the falling particle such as the x & y positions and x & y velocities
    :param data: the tracking data obtained via the TrackPy package
        :param scale: 3-D scale from pixel to mm
    :param frame_rate: frame rate of the video, for converting frame number to time
    :param time_scale: the timescale tau (tau = r/v_light)
    :param diff_method: the method used to differentiate the data
    :param avg_size: the number of neighbors on each side to be averaged
    :param ax: The axis where this figure should be plotted
    :return: motion image at the bottom
    """
    # Create a GridSpec for the 4 plots within the given Axes
    gs_inner = mpl.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=ax, hspace=0.4)

    # Create 4 sub-axes in the provided ax, but use different variable names
    axa = plt.subplot(gs_inner[0])
    axb = plt.subplot(gs_inner[1], sharex=axa)
    axc = plt.subplot(gs_inner[2], sharex=axa)
    axd = plt.subplot(gs_inner[3], sharex=axa)

    # Scale the time, x and y values
    x, y, time = scale_values(data, scale, frame_rate, time_scale)

    # Calculate the x and y velocities
    if diff_method=="five-point":
        deriv_time, vx = five_point_stencil(time, x)
        deriv_time, vy = five_point_stencil(time, y)
    else:
        deriv_time, vx = forward_difference(time, x)
        deriv_time, vy = forward_difference(time, y)
    vy = np.abs(vy)

    # Smoothen the derivative by taking rolling average
    smoothed_vx = rolling_average(vx, neigh_num=avg_size)
    smoothed_vy = rolling_average(vy, neigh_num=avg_size)
    # Calculate average smoothened velocities, ignoring nans
    avg_vx = np.nanmean(smoothed_vx)
    avg_vy = np.nanmean(smoothed_vy)
    # Calculate the standard deviations, ignoring nans
    std_vx = np.nanstd(smoothed_vx)
    std_vy = np.nanstd(smoothed_vy)

    # X displacement plot
    axa.scatter(time, x, marker='+', label='Data')
    axa.plot([time.iloc[0], time.iloc[-1]], [np.mean(x.iloc[:5]), np.mean(x.iloc[-5:])],
             color='g', linestyle='--', label='Drifting')
    axa.set_ylabel("x/R")
    axa.legend()
    axa.label_outer()  # Hide x tick labels

    # Y displacement plot
    axb.scatter(time, -y, marker='+', label='Data')
    axb.plot([time.iloc[0], time.iloc[-1]], [-np.mean(y.iloc[:5]), -np.mean(y.iloc[-5:])],
             color='g', linestyle='--', label='Drifting')
    axb.set_ylabel("y/R")
    axb.legend()
    axb.label_outer()  # Hide x tick labels

    # X velocity plot
    axc.scatter(deriv_time, smoothed_vx, marker='+', label='Smoothed')
    axc.axhline(y=avg_vx, color='r', linestyle='--', label='Average')
    axc.set_ylim(avg_vx-3*std_vx, avg_vx+3*std_vx)
    axc.set_ylabel("$V_x$")
    axc.legend()
    axc.label_outer()  # Hide x tick labels

    # Y velocity plot
    axd.scatter(deriv_time, smoothed_vy, marker='+', label='Smoothed')
    axd.axhline(y=avg_vy, color='r', linestyle='--', label='Average')
    axd.set_ylim(avg_vy-3*std_vy, avg_vy+3*std_vy)
    axd.set_ylabel("$|V_y|$")
    axd.legend()

    # Let the subplots share x-axis
    axd.set_xlabel("Scaled Time")


def plot_stacked_image(frames, ax):
    """
    Plot and return the maximum intensity projection of selected frames in a 3D image stack.

    This function selects 25 evenly spaced slices from the stack and calculates their
    maximum intensity projection (MIP). It then displays this MIP on the provided axes.
    The MIP of the selected frames is also returned for potential use in other functions
    (e.g., for image stretching).

    Parameters
    ----------
    frames : np.ndarray
        3D array representing the image stack, with frames along the first axis.
    ax : matplotlib.axes.Axes
        The axes on which to plot the stacked image.

    Returns
    -------
    mip_selected : np.ndarray
        2D array representing the maximum intensity projection of the selected frames.

    """
    # Number of total frames
    total_frames = frames.shape[0]

    # Calculate the interval at which to take slices
    interval = total_frames // 25

    # Select 25 evenly distributed slices from the stack
    selected_frames = frames[::interval]

    # Maximum Intensity Projection of original selected frames
    mip_selected = np.max(selected_frames, axis=0)

    # Display the stacked image
    ax.imshow(mip_selected, cmap='gray')
    ax.set_title('Stacked Image')
    ax.axis('off')

    return mip_selected  # return this so that it can be used in the stretched image function


def plot_stretched_image(mip_selected, ax):
    """
    Plot a stretched version of an image along the x-axis.

    This function resizes the provided image, extending its width by a factor of 20.
    It then displays this stretched image on the provided axes.

    Parameters
    ----------
    mip_selected : np.ndarray
        2D array representing the image to stretch.
    ax : matplotlib.axes.Axes
        The axes on which to plot the stretched image.

    Returns
    -------
    None

    """
    # Define a new width for the stretched image (e.g., ten times the original width)
    new_width = mip_selected.shape[1] * 20  # change the multiplier as needed

    # Resize the image
    stretched_image = resize(mip_selected, (mip_selected.shape[0], new_width))

    # Display the stretched image
    ax.imshow(stretched_image, cmap='gray', aspect='auto')
    ax.set_title('Stretched Image')
    ax.axis('off')


def compare_2d(frames, particle, thetas, spread):
    """
    This function generates a video that compares the experimental image data and the optimized particle model by frame.
    In each frame of this video, the corrected experimental image will be shown as black and white, while the simulated
    particle shadow will be shown as blue and white.
    :param frames: corrected .tif experimental images
    :param particle: the particle model that will be rotated by optimal theta values recorded in 'thetas'
    :param thetas: optimal theta values found through the optimize_rotation_angle() function
    :param spread: spread of the Gaussian function to determine the pixel intensity of the simulated particle.
    :return: a video that compares the grayscale experimental image and particle model.
    """
    def pad_image_to_macroblock(image, macro_block_size=16):
        """Pad an image to ensure its dimensions are divisible by the given macro_block_size."""
        height, width = image.shape[:2]
        pad_height = (macro_block_size - height % macro_block_size) % macro_block_size
        pad_width = (macro_block_size - width % macro_block_size) % macro_block_size
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        return padded_image

    # Create a writer object
    writer = imageio.get_writer('compare_video/polar_comparison (grayscale).mp4', fps=20)
    shape = np.shape(frames[0])
    for fr in range(len(frames)):
        # Reset the particle and then rotate it by the optimal theta in this frame
        particle.reset()
        particle.rotate('ax2', thetas[fr])
        # Create the shadow of the 3D particle onto the xz plane
        shadow_arr = particle.shadow('xz', shape, find_centroid(frames[fr]), spread=spread)

        # Convert the binary images to 3-channel images
        # Experimental image: white
        img_exp = np.stack([frames[fr], frames[fr], frames[fr]], axis=-1)  # Use frames[fr] to select the current frame

        # Particle model: blue
        img_particle = np.stack(
            [shadow_arr, np.zeros_like(shadow_arr), np.zeros_like(shadow_arr)], axis=-1)

        # Overlap of the experimental image and the particle model: white + blue = cyan
        img_overlap = img_exp + img_particle

        # Concatenate the three images above so that they can be displayed in a row
        img_combined = np.concatenate([img_exp, img_particle, img_overlap], axis=1)

        # Pad the combined image so that its dimension can be divisible by the macro block
        img_padded = pad_image_to_macroblock(img_combined)

        # Write the frame to the video
        writer.append_data(img_padded.astype(np.uint8))

    # Close the writer
    writer.close()


def plot_diagnosis(frames):
    """
    This function plots the diagnostic information for an image sequence. Often times this function is used to check if
    the intensity of the particle is consistent, and whether it may destabilize the modeling algorithm.
    :param frames: The image sequence to be diagnosed.
    :return: A 2 by 1 figure with the top panel showing the total intensity of each frame, and the bottom panel showing
    the number of bright pixels (whose intensity is larger than 0) in each frame.
    """
    # Initialize lists to hold total intensity and bright pixel counts for each frame
    total_intensity = []
    bright_pixels_count = []

    # Iterate over each frame in the sequence
    for frame in frames:
        # Calculate the total intensity of the current frame
        total_intensity.append(np.sum(frame))

        # Count the number of bright pixels in the current frame
        # Here we consider a 'bright pixel' to be any pixel with an intensity > 0
        bright_pixels_count.append(np.sum(frame > 0))

    # Create a 2 by 1 subplot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the total intensity of each frame on the top panel
    axs[0].plot(total_intensity, '-o')
    axs[0].set_title('Total Intensity per Frame')
    axs[0].set_ylabel('Total Intensity')

    # Plot the number of bright pixels of each frame on the bottom panel
    axs[1].plot(bright_pixels_count, '-o', color='orange')
    axs[1].set_title('Number of Bright Pixels per Frame')
    axs[1].set_xlabel('Frame Number')
    axs[1].set_ylabel('Bright Pixels Count')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_rotcurve(time, thetas):
    """
    This function plots the rotation curve of the sedimenting particles, where the rotation angles are found by
    minimizing the differences between experimental images and the particle model.
    :param time: normalized time series.
    :param thetas: optimized rotation angle in radians.
    :return: A matplotlib plot with x-axis being the normalized time and y-axis being the rotation angles in radian.
    """
    # Set the style of the visualization
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))  # Set the size of the figure

    # Plotting the optimal theta values
    plt.plot(time, thetas, marker='o', linestyle='-',
             color=sns.color_palette("viridis", n_colors=3)[0])
    plt.title(r'Optimal $\theta$ over Time', fontsize=16)
    plt.xlabel(r't /$\tau$', fontsize=14)
    plt.ylabel(r'$\theta$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='dotted', alpha=0.7)

    # Set the y-axis limits
    # plt.ylim([0, np.pi])

    # Manually set the tick marks at certain values
    plt.yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
               ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])

    # Show the plot
    plt.tight_layout()
    plt.show()