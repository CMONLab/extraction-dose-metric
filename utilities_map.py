
#!/usr/bin/env python

"""utilities_map.py: Utility functions for creating DSM."""

import numpy as np
from collections import defaultdict

##  Performs linear interpolation on a row of data. ##
def interpolate_row(row, step):
    # Find the index of the first and last non-zero value
    non_zero_indices = np.nonzero(row)[0]
    if len(non_zero_indices) == 0:
        return row
    
    first_non_zero_idx = non_zero_indices[0]
    last_non_zero_idx = non_zero_indices[-1]
    
    # Interpolate between the non-zero values
    for i in range(first_non_zero_idx, last_non_zero_idx + 1):
        if row[i] == 0:
            # Perform linear interpolation
            left_idx = i - 1
            right_idx = i + 1
            
            # Find the nearest non-zero value to the right
            while right_idx <= last_non_zero_idx and row[right_idx] == 0:
                right_idx += 1
                        # Linear interpolation
            if right_idx <= last_non_zero_idx:
                row[i] = row[left_idx] + (row[right_idx] - row[left_idx]) * ((i - left_idx) / (right_idx - left_idx))
    if step == 1:
        return row
    if step == 2:
        if row[-1] == 0:
            row[-1] = row[last_non_zero_idx]
    
    return row

## Replicates rows for even indices. ##
def replication_even(z_map_interpol, indices):
    mid = len(indices) // 2
    for i in range(mid):
        z_map_interpol[indices[i]] = z_map_interpol[indices[i] - 1]
        z_map_interpol[indices[-(i+1)]] = z_map_interpol[indices[-(i+2)]]


## Replicates rows for odd indices. ##
def replication_odd(z_map_interpol, indices):
    mid = len(indices) // 2
        # Replicate the upper row for the first half of the indices
    for i in range(mid):
        z_map_interpol[indices[i]] = z_map_interpol[indices[0] - 1].copy()
    # Replicate the central row with the upper row
    z_map_interpol[indices[mid]] = z_map_interpol[indices[0] - 1].copy()
    # Replicate the lower row for the second half of the indices
    for i in range(mid + 1, len(indices)):
        z_map_interpol[indices[i]] = z_map_interpol[indices[0] + 1].copy()


## Generates a grid for the specified dimensions ##
def generate_grid(bin_x, bin_z):
    xgrid = np.linspace(1, bin_x, bin_x)
    zgrid = np.linspace(1, bin_z, bin_z)
    
    z_index = np.arange(0, bin_x, 1)
    x_index = np.arange(0, bin_z, 1)
    
    total_pixel_def = []
    for i in x_index:
        for j in z_index:
            total_pixel_def.append(((i, j), 0))
            
    return total_pixel_def, xgrid, zgrid


## Map normalization ##
def map_up_to_max(bin_x, bin_z, x_first, z_first, first_dose):
    total_pixel_def, xgrid, zgrid = generate_grid(bin_x, bin_z)
    
    x_p, z_p = np.digitize(x_first, xgrid), np.digitize(z_first, zgrid)
    
    coor_pixel_map = list(zip(z_p, x_p))
    coor_pixel_map_dose = list(zip(coor_pixel_map, first_dose))
    coor_pixel_map_dose_prova = list(zip(z_p, x_p, first_dose))
    unique_slice_z = np.unique(z_p)
    
    max_defi_slice = []
    for i in range(len(unique_slice_z)):
        tmp_max = []
        for a in range(len(coor_pixel_map_dose_prova)):
            if coor_pixel_map_dose_prova[a][0] == unique_slice_z[i]:
                tmp_a = coor_pixel_map_dose_prova[a][1]
                tmp_max.append(tmp_a)
        max_defi_slice.append([i + 1, max(tmp_max)])
            
    max_defi_slice_2 = []
    for i in range(len(max_defi_slice)):
        for a in range(len(coor_pixel_map_dose_prova)):
            if (coor_pixel_map_dose_prova[a][0] == max_defi_slice[i][0] and
                coor_pixel_map_dose_prova[a][1] == max_defi_slice[i][1]):
                max_defi_slice_2.append(coor_pixel_map_dose_prova[a])

    tmp_point_same_pixel = defaultdict(list)
    for k, *v in coor_pixel_map_dose:
        tmp_point_same_pixel[k].append(v)
        
    coor_pixel_map_dose_collapse_point = list(tmp_point_same_pixel.items())
    
    coor_pixel_map_dose_def = []
    for i in range(len(coor_pixel_map_dose_collapse_point)):
        sum_tmp = 0
        tmp_i = np.squeeze(coor_pixel_map_dose_collapse_point[i][1])
        if len(coor_pixel_map_dose_collapse_point[i][1]) > 1:
            sum_tmp = np.round(sum(tmp_i) / len(coor_pixel_map_dose_collapse_point[i][1]), 2)
        else:
            sum_tmp = (np.squeeze(coor_pixel_map_dose_collapse_point[i][1])).tolist()
        coor_pixel_map_dose_def.append((coor_pixel_map_dose_collapse_point[i][0], sum_tmp))

    coor_pixel_map_dose_def_list = [(x, y, z) for (x, y), z in coor_pixel_map_dose_def]
    total_pixel_def_list = [(x, y, z) for (x, y), z in total_pixel_def]
    
    for i in range(len(total_pixel_def_list)):
        for a in range(len(coor_pixel_map_dose_def_list)):
            if (total_pixel_def_list[i][0] == coor_pixel_map_dose_def_list[a][0] and
                total_pixel_def_list[i][1] == coor_pixel_map_dose_def_list[a][1]):
                total_pixel_def_list[i] = coor_pixel_map_dose_def_list[a]
                
    for x in range(len(max_defi_slice_2)):
        for a in range(len(total_pixel_def_list)):
            if (total_pixel_def_list[a][0] == max_defi_slice_2[x][0] and
                total_pixel_def_list[a][1] == bin_x - 1 and
                total_pixel_def_list[a][2] == 0):
                total_pixel_def_list[a] = (total_pixel_def_list[a][0],
                                            total_pixel_def_list[a][1],
                                            max_defi_slice_2[x][2])
                
    z_map_def_ok = [item[2] for item in total_pixel_def_list]
    z_map_def_ok = np.asarray(z_map_def_ok)

    z_map_def_ok_reshape = z_map_def_ok.reshape(bin_z, bin_x)

    # Apply interpolation to each row
    for i in range(z_map_def_ok_reshape.shape[0]):
        z_map_def_ok_reshape[i] = interpolate_row(z_map_def_ok_reshape[i], 2)
                
    return z_map_def_ok_reshape
