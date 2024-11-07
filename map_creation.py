#!/usr/bin/env python

"""map_creation.py: Run script to convert points map to physical map and to normalise """

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from utilities_map import replication_even, replication_odd, interpolate_row, map_up_to_max, generate_grid


def physical_map(x, z, dose):
    """
    Generate a physical dose map using given x, z coordinates and dose values.
    Returns an interpolated map and its corresponding plot.
    """
    z_max, z_min = z.max(), z.min()
    bin_z = int(np.abs(z_max - z_min))
    x_max, x_min = x.max(), x.min()
    bin_x = int(np.abs(x_max - x_min))
    
    # Create grids for x and z
    xgrid = np.linspace(x_min, x_max, bin_x)
    zgrid = np.linspace(z_min, z_max, bin_z)
    
    # Digitize coordinates
    x_p, z_p = np.digitize(x, xgrid), np.digitize(z, zgrid)
    x_p, z_p, dose = x_p.flatten(), z_p.flatten(), dose.flatten()
    
    # Aggregate dose values for each coordinate
    coor_pixel_map_dose = list(zip(zip(z_p, x_p), dose))
    sums_and_counts = defaultdict(lambda: [0, 0])
    for (coords, value) in coor_pixel_map_dose:
        sums_and_counts[coords][0] += value
        sums_and_counts[coords][1] += 1
    collapse_dose_pixel = [((x, y), total / count) for (x, y), (total, count) in sums_and_counts.items()]

    # Create a zero-initialized dose map
    map_physical_space = [((i, j), 0) for i in range(bin_z) for j in range(bin_x)]
    for i in range(len(map_physical_space)):
        for a in range(len(collapse_dose_pixel)):
            if map_physical_space[i][0] == collapse_dose_pixel[a][0]:
                map_physical_space[i] = collapse_dose_pixel[a]

    map_phys_raw = np.array([item[1] for item in map_physical_space]).reshape(bin_z, bin_x)
    plot_matrix, matrix_map_interpol = physical_map_interpolated(map_phys_raw, x_max, x_min, z_max, z_min)
    
    return plot_matrix, matrix_map_interpol, x_max, x_min, z_max, z_min


def physical_map_interpolated(matrix, x_max, x_min, z_max, z_min):
    """
    Interpolates and visualizes a dose map, filling in rows with missing data.
    """
    info_row = []
    for i in range(matrix.shape[0]):
        matrix[i] = interpolate_row(matrix[i], 1)
        if sum(matrix[i]) == 0:
            info_row.append(i)

    info_number_none = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(info_row), lambda ix: ix[0] - ix[1])]
    for idx, val in enumerate(info_number_none):
        if val in ([0], [matrix.shape[0] - 1]):
            continue
        if len(val) == 1:
            matrix[val[0]] = matrix[val[0] - 1] if sum(matrix[val[0] - 1]) != 0 else matrix[val[0] + 1]
        elif len(val) > 1:
            (replication_even if len(val) % 2 == 0 else replication_odd)(matrix, val)

    fig_pixel = plt.imshow(matrix, cmap='turbo', origin='lower', extent=[x_min, x_max, z_min, z_max])
    plt.colorbar(label='Dose [GY]')
    plt.xlabel('X [mm]')
    plt.ylabel('Z [mm]')
    plt.close()
    return fig_pixel, matrix


def normalize_ll_max_row(map_inter, x_max, x_min, z_max, z_min):
    """
    Normalize the dose map by adjusting each row based on its maximum value and generate sub-maps for comparison.
    """
    mean_pos = map_inter.shape[1] // 2
    first_tot_new_pos_x, first_tot_new_pos_y, first_all_dose_tot = [], [], []
    second_tot_new_pos_x, second_tot_new_pos_y, second_all_dose_tot = [], [], []
    
    for i in range(map_inter.shape[0]):
        if not np.all(map_inter[i] == 0):
            # First half of the row
            first_half, second_half = map_inter[i][:mean_pos], map_inter[i][mean_pos:]
            first_indices, second_indices = np.nonzero(first_half)[0], np.nonzero(second_half)[0]
            first_value_dose, second_value_dose = first_half[first_indices], second_half[second_indices]
            percentage_first, percentage_second = mean_pos / len(first_indices) if first_indices.size else 0, mean_pos / len(second_indices) if second_indices.size else 0
            new_first_pos_x = [index * percentage_first for index in range(len(first_indices))]
            new_second_pos_x = [index * percentage_second for index in range(len(second_indices))]
            new_first_pos_y, new_second_pos_y = [i] * len(first_indices), [i] * len(second_indices)
            
            first_tot_new_pos_x.extend(new_first_pos_x)
            first_tot_new_pos_y.extend(new_first_pos_y)
            first_all_dose_tot.extend(first_value_dose)
            second_tot_new_pos_x.extend(new_second_pos_x)
            second_tot_new_pos_y.extend(new_second_pos_y)
            second_all_dose_tot.extend(second_value_dose)
    
    map_1 = map_up_to_max(mean_pos, map_inter.shape[0], first_tot_new_pos_x, first_tot_new_pos_y, first_all_dose_tot)
    map_2 = map_up_to_max(mean_pos, map_inter.shape[0], second_tot_new_pos_x, second_tot_new_pos_y, second_all_dose_tot)
    join_map_max = np.concatenate((map_1, map_2), axis=1)
    join_map_max = np.array([row for row in join_map_max if np.any(row)])

    fig_norm_max = plt.imshow(join_map_max, cmap='turbo', origin='lower', extent=[x_min, x_max, z_min, z_max])
    plt.colorbar(label='Dose [GY]')
    plt.xlabel('X [%]')
    plt.ylabel('Z [%]')
    plt.close()
    return fig_norm_max, join_map_max, map_1, map_2


def normalize_ll(map_1, map_2, prefixed_nx):
    """
    Normalizes maps map_1 and map_2 up to a prefixed length and combines them for comparison.
    """
    nx = int(prefixed_nx.split(',')[0]) // 2
    map_1, map_2 = np.asarray(map_1), np.asarray(map_2)
    
    map_1_up_to_prefixed_length = map_up_to_prefixed_length(map_1, nx)
    map_2_up_to_prefixed_length = map_up_to_prefixed_length(map_2, nx)
    matr_map_up_to_prefixed_length_ok = np.array([row for row in np.concatenate((map_1_up_to_prefixed_length, map_2_up_to_prefixed_length), axis=1) if np.any(row)])

    fig_norm_ll = plt.imshow(matr_map_up_to_prefixed_length_ok, cmap='turbo', origin='lower', extent=[-nx, nx, 0, matr_map_up_to_prefixed_length_ok.shape[0]])
    plt.colorbar(label='Dose [GY]')
    plt.xlabel('X [%]')
    plt.ylabel('Z [%]')
    plt.close()
    return fig_norm_ll, matr_map_up_to_prefixed_length_ok


def map_up_to_prefixed_length(matr_norm, prefixed_nx):
    """
    Adjusts map to a specified length by rescaling pixel positions in x direction up to prefixed_nx.
    """
    percentage_pref_nx = prefixed_nx / matr_norm.shape[1]
    first_tot_new_pos_x, first_tot_new_pos_z, first_all_dose_tot = [], [], []

    for i in range(matr_norm.shape[0]):
        if np.any(matr_norm[i]):
            first_value_dose = matr_norm[i][0:]
            new_first_pos_x = [j * percentage_pref_nx for j in range(matr_norm.shape[1])]
            first_tot_new_pos_x.extend(new_first_pos_x)
            first_tot_new_pos_z.extend([i] * first_value_dose.size)
            first_all_dose_tot.extend(first_value_dose)

    return map_up_to_max(prefixed_nx, matr_norm.shape[0], first_tot_new_pos_x, first_tot_new_pos_z, first_all_dose_tot)


def normalize_cc(matr_norm, norm_parm):
    """
    Normalize and adjust a dose matrix based on parameters for length, height, and fixed parts.
    Expands or compresses the variable part of the matrix to fit the specified dimensions.
    """
    
    # Split the normalization parameters and assign them to length, height, and fixed part values
    norm_parm = norm_parm.split(',')
    length = int(norm_parm[0])
    high = int(norm_parm[1])
    fix = int(norm_parm[2])

    # Separate the fixed part (to be added at the end) and the updatable part
    fix_part = matr_norm[0:fix, :]  # Fixed part to keep unchanged
    up_part = matr_norm[fix:, :]  # Part to stretch or compress
    
    no_fix_map = high - fix  # Remaining height for the adjustable part

    # Calculate the percentage change per row for the stretch/compression of up_part
    perc_no_fix = no_fix_map / up_part.shape[0]

    # Flatten dose values from up_part and prepare coordinate arrays
    dose_up, x_column, z_row = [], [], []
    for j in range(up_part.shape[1]):
        for i in range(up_part.shape[0]):
            dose_up.append(up_part[i][j])

    # Generate x and z coordinate arrays for each column and row
    for i in range(up_part.shape[1]):
        x_column_tmp = [i] * up_part.shape[0]  # X-coordinates (column-wise)
        x_column.append(x_column_tmp)

        z_row_tmp = []
        for a in range(up_part.shape[0]):
            z_row_tmp.append(a * perc_no_fix)  # Z-coordinates adjusted by percentage

        z_row.extend(z_row_tmp)
        x_column_all = [item for sublist in x_column for item in sublist]

    # Define bins for Z and X based on the target dimensions
    bin_z = no_fix_map
    bin_x = up_part.shape[1]
    
    # Generate a grid for the new normalized matrix
    total_pixel_def, xgrid, zgrid = generate_grid(bin_x, bin_z)

    # Digitize the X and Z coordinates for mapping
    x_p, z_p = np.digitize(x_column_all, xgrid), np.digitize(z_row, zgrid)

    # Map each (z_p, x_p) pixel with corresponding dose values
    coor_pixel_map = list(zip(z_p, x_p))
    coor_pixel_map_dose = list(zip(coor_pixel_map, dose_up))

    # Aggregate dose values for pixels in the same location
    tmp_point_same_pixel = defaultdict(list)
    for k, *v in coor_pixel_map_dose:
        tmp_point_same_pixel[k].append(v)

    # Calculate average dose per pixel where multiple doses exist
    coor_pixel_map_dose_collapse_point = list(tmp_point_same_pixel.items())
    coor_pixel_map_dose_def = []
    for i in range(len(coor_pixel_map_dose_collapse_point)):
        sum_tmp = 0
        tmp_i = np.squeeze(coor_pixel_map_dose_collapse_point[i][1])
        if len(coor_pixel_map_dose_collapse_point[i][1]) > 1:
            sum_tmp = np.round(sum(tmp_i) / len(coor_pixel_map_dose_collapse_point[i][1]), 2)
        else:
            sum_tmp = np.squeeze(tmp_i).tolist()
        coor_pixel_map_dose_def.append((coor_pixel_map_dose_collapse_point[i][0], sum_tmp))

    # Extract (X, Y, Z) coordinates from the collapsed dose values
    coor_pixel_map_dose_def_list = [(x, y, z) for (x, y), z in coor_pixel_map_dose_def]

    # If up_part is smaller than the target map, expand it
    if up_part.shape[0] < no_fix_map:
        after_point_collapse_pixel = 0
        for i in range(len(total_pixel_def)):
            for a in range(len(coor_pixel_map_dose_def)):
                if total_pixel_def[i][0] == coor_pixel_map_dose_def[a][0]:
                    after_point_collapse_pixel += 1
                    total_pixel_def[i] = coor_pixel_map_dose_def[a]
                    
        z_map_def_ok_3 = [item[1] for item in total_pixel_def]
        z_map_def_ok_3 = np.asarray(z_map_def_ok_3)
        z_map_def_ok_reshape = z_map_def_ok_3.reshape(bin_z, bin_x)

        # Interpolate the new matrix for smoother output
        fig_map_no_fix, matrix_no_fix = physical_map_interpolated(z_map_def_ok_reshape, 0, z_map_def_ok_reshape.shape[0], 0, z_map_def_ok_reshape.shape[1])
        
        # Concatenate the fixed part and the newly adjusted matrix
        matr_all_cc = np.concatenate((fix_part, matrix_no_fix), axis=0)
        map_cc_plot = plt.imshow(matr_all_cc, cmap='turbo', origin='lower', extent=[-(length / 2), length / 2, 0, high])
        bar13_plot = plt.colorbar()
        bar13_plot.set_label('Dose [GY]')
        plt.xlabel('X [%]')
        plt.ylabel('Z [%]')
        plt.close()
        plt.clf()
    
    # If up_part is larger than the target map, compress it
    if up_part.shape[0] >= no_fix_map:
        z_map_def_ok = [item[2] for item in coor_pixel_map_dose_def_list]
        z_map_def_ok2 = [z_map_def_ok[n:n + bin_z] for n in range(0, len(z_map_def_ok), bin_z)]
        z_map_def_ok2 = np.asarray(z_map_def_ok2).transpose()

        # Concatenate the fixed part with the compressed upper part
        matr_all_cc = np.concatenate((fix_part, z_map_def_ok2), axis=0)
        map_cc_plot = plt.imshow(matr_all_cc, cmap='turbo', origin='lower', extent=[-(length / 2), length / 2, 0, high])
        bar12_plot = plt.colorbar()
        bar12_plot.set_label('Dose [GY]')
        plt.xlabel('X [%]')
        plt.ylabel('Z [%]')
        plt.close()
        plt.clf()

    return map_cc_plot, matr_all_cc
