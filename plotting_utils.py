'''
Plotting functions used to generate figures in the paper.
'''

from matplotlib import colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import fourier_transform_utils as ftu
import numpy as np
import dataframe_image as dfi
import pandas as pd
import mrcfile
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
names_to_show = { "diagonal": "diagonal", "wilson": "Wilson", "diagonal masked": "diagonal masked", "wilson masked": "Wilson masked" }
colors_name = { "diagonal": "cornflowerblue", "wilson": "lightsalmon", "diagonal masked": "blue", "wilson masked": "orangered"  }

### PLOTTING STUFF
def FSC(image1, image2, r_dict = None):
    # Old verison from me:
    r_dict = ftu.compute_index_dict(image1.shape) if r_dict is None else r_dict
    top_img = image1 * np.conj(image2)
    top = ftu.compute_spherical_average(top_img, r_dict)
    if np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top)) > 1e-6:
        print("FDC not real. Normalized error:", np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top)))
    top = np.real(top)
    bot = np.sqrt(ftu.compute_spherical_average(np.abs(image1)**2, r_dict) * ftu.compute_spherical_average(np.abs(image2)**2, r_dict) )
    bin_fsc = top / bot
    return bin_fsc

def fsc_score(fsc_curve, grid_size, voxel_size, threshold = 0.5 ):
    # First index below 0.5
    freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
    freq = freq[freq >= 0 ]
    idx = int(np.max(np.argmin(fsc_curve >= threshold), 0))
    return freq[idx]


def plot_all_images_for_paper(images_to_plot, n_dirs, global_name, scale_image_name, voxel_size, save_to_file):
    for (key, image) in images_to_plot.items():
        if save_to_file:
            with mrcfile.new(global_name + key + ".mrc", overwrite=True) as mrc:
                mrc.set_data(image.astype(np.float32))

    for idx in range(n_dirs):
        projections = {}
        for (key, image) in images_to_plot.items():
            projections[key] = np.real(np.sum(image, axis = idx))
        plot_images_on_same_scale(projections, global_name + "proj" + str(idx), scale_image_name,voxel_size, save_to_file)

        slices = {}
        for (key, image) in images_to_plot.items():
            slices[key] = np.real(image.take(image.shape[0]//2, axis = idx))
        plot_images_on_same_scale(slices, global_name + "slice" + str(idx), scale_image_name, voxel_size,save_to_file)

def plot_images_on_same_scale(images_to_plot, global_name, scale_image_name,  voxel_size, save_to_file):
    min_val_all = np.inf
    max_val_all = - np.inf
    for img in images_to_plot.values():
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    for (name, image) in images_to_plot.items():
        plot_function_with_scale(image, global_name + name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = False)
    if scale_image_name in images_to_plot:
        plot_function_with_scale(images_to_plot[scale_image_name],  global_name + scale_image_name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = True)
    return
    

def plot_function_with_scale(image, name, vmin, vmax, voxel_size, save_to_file = False, show_colorbar = False, show_scalebar = True):
        # Create subplot
        fig, ax = plt.subplots()
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
        
        if show_scalebar:
            # Create scale bar
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)

        if show_colorbar:
            fig.colorbar(pos, ax = ax)
        if save_to_file:
            plt.savefig(name + ".pdf", bbox_inches='tight')

def plot_fsc_function_paper(fsc_curves, global_name, names, grid_size, voxel_size, save_to_file = False):
    plot_fsc_flag = True
    if plot_fsc_flag:
        plt.figure(figsize=(9, 8))
        ax = plt.gca()
        plt.rcParams['text.usetex'] = False
        freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
        freq = freq[freq >= 0 ]
        freq = freq[:grid_size//2 ]

        lines = []
        line_names = []
        colors_plotted  = []
        names_plotted  = []
        scores = {}
        for name in names:
            curve = fsc_curves[name]
            max_idx = min(curve.size, freq.size)
            plt.plot(freq[:max_idx], curve[:max_idx], color = colors[colors_name[name]] , label = names_to_show[name] , linewidth = 2 )
            colors_plotted.append(colors[colors_name[name]])
            names_plotted.append(names_to_show[name])

            curve = fsc_curves[name + " masked"]
            max_idx = min(curve.size, freq.size)
            line = plt.plot(freq[:max_idx], curve[:max_idx],color =  colors[colors_name[name + " masked"]], label = names_to_show[name + " masked"], linewidth = 2 )
            colors_plotted.append(colors[colors_name[name +" masked"]])
            names_plotted.append(names_to_show[name + " masked"])

            scores[name] = fsc_score(curve, grid_size, voxel_size)

        n_dots_in_line = 20
        for name in names:
            plt.plot(np.ones(n_dots_in_line) * scores[name], np.linspace(0,1, n_dots_in_line), "k-")

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        plt.plot(freq, 0.5 * np.ones(freq.size), "k--")
        plt.ylim([0, 1.02])
        plt.xlim([0, np.max(freq)])
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=20) 

        if save_to_file:
            global_name = global_name + "fsc.pdf"
            plt.savefig(global_name, bbox_inches='tight')

        plt.figure()
        f = lambda m,c: plt.plot([],[], color=c, ls="-")[0]
        handles = [f("-", colors_plotted[i]) for i in range(4)]
        labels = names_plotted
        legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True, prop={'size': 20})
        if save_to_file:
            export_legend(legend, filename = global_name + "legend.pdf")
        plt.show()



def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def make_images_for_paper(denoised_images, clean_image, noisy_image, mask, grid_size, voxel_size, paper_save_directory, experiment_name, log_SNR, save_to_file):
        
    global_name = paper_save_directory  + experiment_name + "_" + str(log_SNR)

    from fourier_transform_utils import get_inverse_fourier_transform, get_fourier_transform
    ### BELOW THIS IS PLOTTING.
    fsc_curves = {};   resolutions = {}
    fsc_curves_masked = {}; 
    names_to_plot = ["diagonal", "wilson"]
    clean_image_ft = get_fourier_transform(clean_image, voxel_size)

    for name in names_to_plot:
        denoised_images_ft = get_fourier_transform(denoised_images[name], voxel_size)
        fsc_curves[name] = FSC(denoised_images_ft, clean_image_ft)

        denoised_masked_ft = get_fourier_transform(denoised_images[name] * mask, voxel_size)
        fsc_curves_masked[name] = FSC(denoised_masked_ft,clean_image_ft)

        score = fsc_score(fsc_curves_masked[name], grid_size, voxel_size)
        resolutions[name] = 1/score


    names_in_denoised = ["diagonal", "wilson"]
    images_to_plot = {}
    for name in names_in_denoised:
        images_to_plot[name] = denoised_images[name]
    images_to_plot["clean"] = clean_image

    plot_all_images_for_paper(images_to_plot, 1,  global_name, scale_image_name = "clean", voxel_size = voxel_size, save_to_file=save_to_file)
    plot_all_images_for_paper({"noisy": noisy_image}, 1, global_name,  scale_image_name = "noisy",voxel_size = voxel_size, save_to_file=save_to_file)


    fsc_curves_to_plot = {}
    plot_names = { "diagonal": "diagonal", "wilson" : "wilson" }

    for name in plot_names:
        fsc_curves_to_plot[plot_names[name]] = fsc_curves[name]

    for name in plot_names:
        fsc_curves_to_plot[plot_names[name] + " masked"] = fsc_curves_masked[name]

    plot_fsc_function_paper(fsc_curves_to_plot , global_name, names_in_denoised, grid_size, voxel_size,   save_to_file)


    kk = pd.DataFrame(resolutions, ["resolution"])
    score_df = kk.T
    score_df
    df_styled = score_df.style.highlight_min(subset = ["resolution"], color = "green")
    #plt.savefig('mytable.png')
    display(df_styled)
    if save_to_file:
        dfi.export(df_styled, paper_save_directory + experiment_name + "scores_table"+ str(log_SNR) + ".png")


# Plotting for sample figures
def plot_samples_on_same_scale(images_to_plot, save_filepath, voxel_size, save_to_file):
    min_val_all = 0
    max_val_all = -np.inf
    for img in images_to_plot:
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    idx = 0 
    for image in images_to_plot:
        plot_sample_with_scale(image, save_filepath + str(idx), voxel_size, min_val_all, max_val_all, save_to_file = save_to_file, show_colorbar = True)
        idx +=1 
    return

def plot_sample_with_scale(image, save_filepath, voxel_size, vmin , vmax , save_to_file = False, show_colorbar = True, show_scalebar = True):
        fig, ax = plt.subplots(figsize = (4,4))
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
    
        if show_scalebar:
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)
        plt.show()

        if save_to_file:
            plt.savefig(save_filepath, bbox_inches='tight')

        if show_colorbar:
            fig, ax = plt.subplots(figsize = (4,4))
            plt.gca().set_visible(False)
            cbar = fig.colorbar(pos, orientation="horizontal")
            cbar.ax.tick_params(labelsize=14)
            plt.savefig(save_filepath + "colorbar" , bbox_inches='tight')
