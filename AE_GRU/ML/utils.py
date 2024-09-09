import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS.keys())


def create_image_grid(image_paths, grid_size=(2, 5)):
    """
    Create a grid of images from the provided list of image paths without resizing the images.

    Parameters:
    - image_paths: List of paths to the images.
    - grid_size: Tuple indicating the number of rows and columns in the grid.

    Returns:
    - An Image object representing the grid of images.
    """
    # Open the first image to determine the size
    with Image.open(image_paths[0]) as img:
        image_size = img.size

    # Determine the size of the grid
    grid_width = grid_size[1] * image_size[0]
    grid_height = grid_size[0] * image_size[1]

    # Create a new image with white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Iterate over each image and paste it into the grid
    for index, image_path in enumerate(image_paths):
        # Open the image
        with Image.open(image_path) as image:
            # Calculate the position where the image will be pasted
            x_position = (index % grid_size[1]) * image_size[0]
            y_position = (index // grid_size[1]) * image_size[1]

            # Paste the image onto the grid
            grid_image.paste(image, (x_position, y_position))

    return grid_image


def plot_obs_prediction(estm_traj, real_traj, title="", show=False):
    # Create a figure and a grid of subplots
    fig = plt.figure(figsize=(5, real_traj.shape[0]*1.5))
    
    if estm_traj.shape[0] == 1:
        ts = np.arange(estm_traj[0, :].shape[-1])/12
        ax = fig.add_subplot()
        ax.set(title=title, xlabel='Time (Hour)', xticks=np.arange(0, ts[-1], step=24),
               ylabel='Actigraphy (a.u.)')
        # ax.set(xticks=np.arange(0, ts[-1], step=24))
        # ax.set_xlabel('Time (Hour)', fontsize=16)
        # ax.set_ylabel('Actigraphy (a.u.)', fontsize=16)
        # ax.set_title(title, fontsize=16)
        ax.plot(ts, estm_traj[0, :], label='Estimate', linewidth=2, alpha=0.6)
        ax.plot(ts, real_traj[0, :], label='Real', linewidth=2, alpha=0.3)
        ax.legend(loc=1)
    else:
        for i in range(real_traj.shape[0]):

            ax = fig.add_subplot(real_traj.shape[0]//2, 2, i+1)
            ax.plot(estm_traj[i, :], label='Estimate', linewidth=1, alpha=0.6)
            ax.plot(real_traj[i, :], label='Real', linewidth=1, alpha=0.3)
            # ax.set_title(f"$o_{i}$")
            if i == 0:
                ax.set(title=title)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if show:
        plt.show()
        plt.close(fig)
    return fig
    


    
def plot_state_trajectory(estm_traj, inputs, real_traj=None, 
                          title="", label=None, xlim=None, ylim=None,
                          show=False):
    
    colors = []
    if label is not None:
        for l in label:
            colors.append(color_list[l % len(color_list)])
    else:
        colors = [color_list[0] for _ in range(estm_traj.shape[-1])]

    fontsize = 16
    
    # Create a figure and a grid of subplots
    fig = plt.figure(figsize=(5, 3))

    ts = np.arange(estm_traj[0, :].shape[-1])/12
    xticks = np.arange(0, ts[-1], step=24)

    # Create the subplot for the left column, top position (1st figure)
    ax1 = fig.add_subplot(3, 2, 1)  # 2x2 grid, 1st subplot
    ax1.plot(ts, estm_traj[0, :], label='Estimate', linewidth=3)
    if real_traj is not None:
        ax1.plot(ts, real_traj[0, :], label='Real', linewidth=1)
    ax1.set_ylabel("$x_1$", fontsize=fontsize)
    ax1.set(xticks=xticks)

    # Create the subplot for the left column, bottom position (2nd figure)
    ax2 = fig.add_subplot(3, 2, 3)  # 2x2 grid, 3rd subplot
    ax2.plot(ts, estm_traj[1, :], label='Estimate', linewidth=3)
    if real_traj is not None:
        ax2.plot(ts, real_traj[1, :], label='Real', linewidth=1)
    ax2.set_ylabel("$x_2$", fontsize=fontsize)
    ax2.set(xticks=xticks)
    
    ax4 = fig.add_subplot(3, 2, 5)  # 2x2 grid, 3rd subplot
    ax4.plot(ts, inputs.flatten()[-ts.shape[-1]:], label='input', linewidth=1, color='tab:orange')
    ax4.set_ylabel("$u$", fontsize=fontsize)
    ax4.set_xlabel("Time (Hour)", fontsize=fontsize)
    ax4.set(xticks=xticks)


    # Create the subplot for the right column (3rd figure)
    ax3 = fig.add_subplot(1, 2, 2)  # 1x2 grid, 2nd subplot
    ax3.plot(estm_traj[0, :], estm_traj[1, :], color='k', label='Estimate', linewidth=1, alpha=0.3)
    
    for i in range(estm_traj.shape[-1]):
        ax3.plot(estm_traj[0, i], estm_traj[1, i], marker='o', color=colors[i], markersize=3, alpha=0.3)
    # if real_traj is not None:
    # ax3.plot(real_traj[0, :], real_traj[1, :], color='tab:orange', label='Real', linewidth=1)
    ax3.set_title(title, fontsize=fontsize)
    
    ax3.scatter(estm_traj[0, 0], estm_traj[1, 0], color='tab:green', marker='X', label='Estimate', s=100)
    ax3.scatter(estm_traj[0, -1], estm_traj[1, -1], color='tab:red', marker='X', label='Estimate', s=100)
    # ax3.plot(real_traj[0, 0], real_traj[1, 0], color='tab:orange', marker='x', label='Real')
    # ax3.plot(real_traj[0, -1], real_traj[1, -1], color='tab:orange', marker='o', label='Real')
    ax3.set(xlim=xlim, ylim=ylim)
    ax3.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)

    return fig

    
def plot_trajectory_gif(x, y):
    
    fig, ax = plt.subplots()
    # Setting the limits of x and y axes
    ax.set_xlim(np.min(x) - 0.1, np.max(x) + 0.1)
    ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)

    # Initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # Animation function: this is called sequentially
    def animate(i):
        line.set_data(x[:i+1], y[:i+1])
        return line,
    
    # Initialize an empty plot
    line, = ax.plot([], [], 'bo-', lw=2, alpha=0.3)
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=200, blit=True)
    ani.save('trajectory_animation.gif', writer='imagemagick')
