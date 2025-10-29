import os
import matplotlib.pyplot as plt
import numpy as np

def create_horizontal_colorbar(colormap, cbar_min, cbar_max, label, filename, step):
    # Create a ScalarMappable to map normalized data to colors
    step_size = (cbar_max - cbar_min) / step
    norm = plt.Normalize(cbar_min, cbar_max)
    increment = step_size
    cmap = colormap

    # Create a figure for the colorbar
    fig, ax = plt.subplots(figsize=(3.25, 1.5))

    # Draw the colorbar
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation='horizontal',
        ticks=np.arange(cbar_min, cbar_max + increment, increment)
    )
    cbar.set_label(label, rotation=0, labelpad=10, fontsize=10)
    cbar.ax.xaxis.set_label_coords(0.5, -1.5)

    # Align the tick labels to the bottom
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_va('top')

    cbar.ax.tick_params(labelsize=7, pad=5, width=0.25)

    # Make the border thinner
    cbar.outline.set_linewidth(0.25)

    # Remove the axis
    ax.axis('off')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(filename, format='svg', dpi=600, transparent=True)
    plt.show()

if __name__ == '__main__':
    output_folder_path = "Colorbars_for_Figures/"
    if os.path.exists(output_folder_path) == False:
        os.makedirs(output_folder_path)
    
    create_horizontal_colorbar(
        colormap=plt.cm.Greys,
        cbar_min=0,
        cbar_max=4,
        label=r'$\Delta L/D$ Error',
        filename=os.path.join(output_folder_path, 'Error_delta_L_D_colorbar_horizontal.svg'),
        step=4
    )

    create_horizontal_colorbar(
        colormap=plt.cm.Blues,
        cbar_min=0,
        cbar_max=2,
        label=r'$RMSE$',
        filename=os.path.join(output_folder_path, 'RMSE_colorbar_horizontal.svg'),
        step=4
    )

    create_horizontal_colorbar(
        colormap=plt.cm.coolwarm,
        cbar_min=6,
        cbar_max=-6,
        label=r'$\Delta L/D$',
        filename=os.path.join(output_folder_path, 'Delta_L_D_colorbar_horizontal.svg'),
        step=6
    )

    create_horizontal_colorbar(
        colormap=plt.cm.PuOr_r,
        # cbar_min=-134,
        # cbar_max=134,
        cbar_min=-248.84,
        cbar_max=248.84,
        label=r'$\Delta p$\n(Pa)',
        filename=os.path.join(output_folder_path, 'Delta_p_colorbar_horizontal.svg'),
        step=6
    )