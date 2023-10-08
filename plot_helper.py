import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from skimage.transform import resize
from data_handler import *
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
        coordinates), center_of_geometry (list of coordinates), offset, theta, and phi.
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


def plot_shadow(particle, domain_size, scale=7):
    """
    Visualize the shadow of the particle on both 'xz' and 'yz' planes.

    Parameters:
    - particle: The Particle object.
    - domain_size: Side length of the squared domain in mm.
    - scale: The scale factor converting mm to pixels (default is 7).
    """
    # Generate the shadow grids
    shadow_grid_xz = particle.shadow('xz', domain_size, scale)
    shadow_grid_yz = particle.shadow('yz', domain_size, scale)

    # Create a plot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    # Display the shadow grid for 'xz' plane
    axs[0].imshow(shadow_grid_xz, cmap='gray_r', aspect='equal')
    axs[0].axis('off')
    axs[0].set_title('Shadow on X-Z Plane')

    # Display the shadow grid for 'yz' plane
    axs[1].imshow(shadow_grid_yz, cmap='gray_r', aspect='equal')
    axs[1].axis('off')
    axs[1].set_title('Shadow on Y-Z Plane')

    # Display the plot
    plt.tight_layout()
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