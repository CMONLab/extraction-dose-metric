#!/usr/bin/env python

"""dsh_algorithm.py: This script processes dose data and STL files to calculate dose surface map."""

import pydicom as py
import os
import sys
import numpy as np
import pandas as pd
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utilities import find_file_with_string



def calculate_dsh(dose_file, arg_str, stl_file, bin):

    """
    Args:
        dose_file: Path to the DICOM dose file.
        arg_str: Search string for finding the STL file.
        stl_file: List of STL file paths to search.
        bin: Number of bins for dose histogram.

    Returns:
        DataFrame containing dose statistics.
    """
    
    selected_mesh = find_file_with_string(arg_str, stl_file)

    # Check if the selected mesh is found
    if selected_mesh is None:
        print(f"Warning: No file found in 'stl_file' containing the string '{arg_str}'")
        return None

    print('Selected mesh:', selected_mesh)
    DAT_dsh, DAT_value_dsh = [], []
    
    # Load the dose data from the DICOM file
    dose_file = py.read_file(dose_file, force=True)
    patient_name = dose_file.PatientID

    # Get the STL file directory path and mesh
    dir_path_stl = os.path.dirname(os.path.realpath(selected_mesh))
    string_stl = os.path.basename(os.path.normpath(selected_mesh))
    
    my_mesh = mesh.Mesh.from_file(selected_mesh)
    mesh_points = my_mesh.points
    
    # Extract dose matrix and transpose for processing
    matrix_pixel = (dose_file.pixel_array) * dose_file.DoseGridScaling
    matrix_dose_T = np.transpose(matrix_pixel)
    
    # Get dose file coordinates
    rows = dose_file.Rows
    columns = dose_file.Columns
    pixel_spacing = dose_file.PixelSpacing
    image_position = dose_file.ImagePositionPatient
    
    # Compute new coordinate axes based on pixel spacing
    x_axis_new = np.arange(columns) * pixel_spacing[0] + image_position[0]
    y_axis_new = np.arange(rows) * pixel_spacing[1] + image_position[1]
    z_axis_new = np.array(dose_file.GridFrameOffsetVector) + image_position[2]
    
    z_spacing = dose_file.GridFrameOffsetVector[1]
    # Reshape mesh points for easier coordinate handling
    mesh_points_reshape = mesh_points.reshape(-1)
    dim_to_reshape = int(mesh_points_reshape.shape[0] / 3)
    mesh_points_ok = mesh_points_reshape.reshape(dim_to_reshape, 3)
    
    # Extract mesh coordinates
    x_mesh = mesh_points_ok[:, 0]
    y_mesh = mesh_points_ok[:, 1]
    z_mesh = mesh_points_ok[:, 2]
  
    # Convert mesh coordinates to pixel coordinates
    x_coord_pix = np.ceil((x_mesh - image_position[0]) / pixel_spacing[0]).astype(int)
    y_coord_pix = np.ceil((y_mesh - image_position[1]) / pixel_spacing[1]).astype(int)
    z_coord_pix = np.ceil((z_mesh - image_position[2]) / z_spacing).astype(int)

    # Gather dose values corresponding to mesh points
    value_dose = []
    for c in range(len(x_coord_pix)):
        value_dose.append(matrix_dose_T[x_coord_pix[c], y_coord_pix[c], z_coord_pix[c]])


    value_dose = np.array(value_dose)

    # Calculate max dose value
    max_value_dose = np.max(value_dose)
    dim_triangle = dim_to_reshape * 3
    area_array_triangle = my_mesh.areas
    value_dose_mean = np.mean(value_dose.reshape(-1, 3), axis=1)
    area_array_triangle_tmp = area_array_triangle.reshape(-1)

    # Calculate total area of the mesh
    area_tot_mesh = np.sum(area_array_triangle_tmp)
    print("Total area of the mesh:", area_tot_mesh)

    # Prepare bins for histogram
    dim_bin = np.arange(0, np.max(value_dose_mean), 0.5)
    dose_super = np.array(list(zip(value_dose_mean, area_array_triangle_tmp)))
    dose_super_sort = sorted(dose_super, key=lambda x: x[0])

    vect_size_dose = np.zeros(len(dim_bin))  # Initialize vector for dose accumulation
    for bin_index in range(len(dim_bin)):
        vect_tmp = sum(area for value, area in dose_super_sort if value > dim_bin[bin_index])
        vect_size_dose[bin_index] = vect_tmp

    # Normalize and prepare for plotting
    vect_size_dose = vect_size_dose[1:]
    max_vect_size_dose = np.max(vect_size_dose)
    vect_value_dsh = vect_size_dose * (float(bin) / max_vect_size_dose)  # Normalize the vector based on the max

    # Prepare the data for saving
    DAT_dsh.append([patient_name, string_stl, np.round(area_tot_mesh, 2), np.round(max_value_dose, 2)])
    DAT_value_dsh = np.round(vect_value_dsh.reshape(1, -1), 2)

    # Create DataFrames for results
    df1_dsh = pd.DataFrame(DAT_dsh)
    df2_dsh = pd.DataFrame(DAT_value_dsh)
    result = pd.concat([df1_dsh, df2_dsh], axis=1)
    
    return result  # Return the resulting DataFrame

