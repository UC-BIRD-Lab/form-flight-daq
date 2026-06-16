import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def create_horizontal_colorbar(colormap, cbar_min, cbar_max, label, filename, step):
    cmap = colormap
    vmin, vmax = cbar_min, cbar_max
    if vmin > vmax:
        vmin, vmax = vmax, vmin
        cmap = colormap

    # symmetric center at 0 if range crosses 0 
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # evenly spaced ticks (step = number of intervals)
    ticks = np.linspace(vmin, vmax, step + 1)

    fig, ax = plt.subplots(figsize=(3.25, 1.5))
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, orientation='horizontal', ticks=np.round(ticks, 1)
    )

    cbar.set_label(label, rotation=0, labelpad=10, fontsize=10)
    cbar.ax.xaxis.set_label_coords(0.5, -1.5)

    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_va('top')

    cbar.ax.tick_params(labelsize=7, pad=5, width=0.25)
    cbar.outline.set_linewidth(0.25)
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
        cbar_max=1.7, # Set maximum value for color scale for RMSE heatmap, the max is 1.67, so I set to 1.7
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

    teal_white_coral = LinearSegmentedColormap.from_list(
        'amber_white_teal', ['#c87d0a', '#ffffff', '#0a7e6e']
    )
    create_horizontal_colorbar(
        colormap=teal_white_coral,
        cbar_min=-4,
        cbar_max=4,
        label=r'Prediction error ($\hat{y} - y$)',
        filename=os.path.join(output_folder_path, 'XGB_Error_colorbar_horizontal.svg'),
        step=4
    )