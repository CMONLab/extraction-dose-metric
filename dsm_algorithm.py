#!/usr/bin/env python

"""dsm_algorithm.py: Script to calculate DSM."""

import os
import pydicom as py
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from dicompylercore import dicomparser
from scipy.interpolate import splprep, splev
from utilities import save_img, spline_interpolation, is_clockwise, separate_positive_negative_values, find_nearest_point, sort_points_by_greedy_distance, reorder_dose, cumulative_distances, check_concavity
from scipy.spatial import distance
from map_creation import physical_map, normalize_ll_max_row, normalize_cc, normalize_ll
import csv
from shapely.geometry import LineString, MultiPoint
import time

# Define the color map for plotting
cmap_turbo = plt.get_cmap('turbo')

def calculate_dsm(file_struct, file_dose, matrix_dose_eqd2, structure_name, cut_open, norm_param, start_time):
    # Get the path to save output files
    path_to_save = os.path.dirname(file_struct)
    # Read DICOM structure and dose files
    data_struct = py.read_file(file_struct, force=True)
    data_dose = py.read_file(file_dose, force=True)

    patient_name = data_dose.PatientID
    structure_name = structure_name.split(',')

    pixel_spacing = data_dose.PixelSpacing
    image_position = data_dose.ImagePositionPatient
    z_spacing = data_dose.GridFrameOffsetVector[1]

    # Parse the DICOM structure set
    RTss = dicomparser.DicomParser(file_struct)
    RTstructures = RTss.GetStructures()
    
    all_name_str = []
    info_struct = os.path.join(path_to_save, 'pts_name_struct.txt')
    
    with open(info_struct, 'w') as out_file:
        out_file.write("Name_struct \n")

        structure_found_global = False
        for key, structure in RTstructures.items():
            out_file.write(structure['name'] + '\n')
            all_name_str.append(structure['name'])
            
            structure_found = False
            for s in structure_name:
                if re.match(s, structure['name'], re.IGNORECASE):
                    structure_found = True
                    structure_found_global = True

                    print("structure name:", structure['name'])
                    x_arg, y_arg, z_arg = [], [], []

                    # Check the concavity of the structure
                    result_concavity = check_concavity(RTss.GetStructureCoordinates(key).items())
                    
                    if result_concavity == True:
                        # Structure is valid and has no concavity issues
                        matrix_slide_vescica_json = [structure2[0]['data'] for key2, structure2 in RTss.GetStructureCoordinates(key).items()]
                        report_issue = 0

                    elif result_concavity == False:
                        # Structure has problems: z_spacing_slice is not consecutive
                        report_issue = 1
                        continue  # Skip this iteration and continue with the next structure

                    elif isinstance(result_concavity, tuple):
                        # At least 2 slices are needed to create the structure map
                        part_to_extract_map, pos = result_concavity
                        report_issue = f"{2}, {pos}"
                        keys_concavity = [str(key) for key in part_to_extract_map]
                        keys_to_keep_str = [f"{float(key):.2f}" for key in keys_concavity]
                        data_dict = dict(RTss.GetStructureCoordinates(key).items())
                        filtered_dict = {key: value for key, value in data_dict.items() if key in keys_to_keep_str}
                        matrix_slide_vescica_json = [structure2[0]['data'] for key2, structure2 in filtered_dict.items()]

                    if len(matrix_slide_vescica_json) >= 2:
                        matrix_slide_vescica = np.vstack(matrix_slide_vescica_json)
                        x_arg, y_arg, z_arg = matrix_slide_vescica[:, 0], matrix_slide_vescica[:, 1], matrix_slide_vescica[:, 2]

                        # Calculate pixel coordinates based on DICOM image position and pixel spacing
                        x_coord_pix = np.ceil((x_arg - image_position[0]) / pixel_spacing[0]).astype(int)
                        y_coord_pix = np.ceil((y_arg - image_position[1]) / pixel_spacing[1]).astype(int)
                        z_coord_pix = np.ceil((z_arg - image_position[2]) / z_spacing).astype(int)
                        value_dose = matrix_dose_eqd2[x_coord_pix, y_coord_pix, z_coord_pix]

                        # Translate the slice into a 2D representation
                        plot_points, unfolding_slice_x, unfolding_slice_z, unfolding_slice_dose = translate_slice(x_arg, y_arg, z_arg, value_dose, matrix_dose_eqd2, image_position, pixel_spacing, z_spacing, cut_open)
                        save_img(plot_points, path_to_save, patient_name, structure['name'], '_map_points_unfold_', cut_open, None)
                        
                        # Create and save various maps
                        plot_matrix, matrix_map, x_max, x_min, z_max, z_min = physical_map(unfolding_slice_x, unfolding_slice_z, unfolding_slice_dose)
                        save_img(plot_matrix, path_to_save, patient_name, structure['name'], '_map_pixel_raw_', cut_open, matrix_map)
                        
                        plot_map_ll_max_row, matrix_map_norm_ll_max_row, map_1, map_2 = normalize_ll_max_row(matrix_map, x_max, x_min, z_max, z_min)
                        save_img(plot_map_ll_max_row, path_to_save, patient_name, structure['name'], '_map_pixel_ll_extend_', cut_open, matrix_map_norm_ll_max_row)
                        
                        plot_map_ll, matrix_map_norm_ll = normalize_ll(map_1, map_2, norm_param)
                        save_img(plot_map_ll, path_to_save, patient_name, structure['name'], '_map_pixel_ll_max_slice_', cut_open, matrix_map_norm_ll)
                        
                        end_time = time.time()
                        execution_time = end_time - start_time

                        print(f"Execution time: {execution_time} seconds")

                        # To activate CC normalization
                        plot_map_cc, matrix_map_norm_cc = normalize_cc(matrix_map_norm_ll, norm_param)
                        save_img(plot_map_cc, path_to_save, patient_name, structure['name'], '_map_pixel_norm_cc_', cut_open, matrix_map_norm_cc)

                    else:
                        continue  # Skip this structure if it has fewer than 2 slices

    if not structure_found_global:
        report_issue = 'no structure found'
    
    return patient_name, report_issue
                                

def translate_slice(x, y, z, dose_point_contour, dose_matrix, image_position, pixel_spacing, z_spacing, cut_open):
    # Count the occurrences of each z coordinate
    num_repeat_same_z = Counter(z)
    num_repeat_values = np.array(list(num_repeat_same_z.values()))
    num_repeat_values = np.insert(num_repeat_values, 0, 0)  # Insert 0 at the beginning

    # Compute cumulative sum to get the indices of points for each slice
    number_point_slice = np.cumsum(num_repeat_values).tolist()

    # Lists to store unfolded slice data
    unfolding_slice_x, unfolding_slice_z, unfolding_slice_dose = [], [], []

    # Iterate over each slice defined by the number of points
    for i in range(len(number_point_slice) - 1):
        # Check if there are at least 6 points for the slice
        if number_point_slice[i + 1] > 5:
            # Extract points for the current slice
            slice_x_esima = x[number_point_slice[i]:number_point_slice[i + 1]]
            slice_y_esima = y[number_point_slice[i]:number_point_slice[i + 1]]
            slice_z_esima = z[number_point_slice[i]:number_point_slice[i + 1]]
            slice_dose_esima = dose_point_contour[number_point_slice[i]:number_point_slice[i + 1]]

            # Calculate the centroid of the slice
            centroid_x, centroid_y = np.mean(slice_x_esima), np.mean(slice_y_esima)
            # Translate the points to center around the centroid
            slice_x_trasl_esima = slice_x_esima - centroid_x
            slice_y_trasl_esima = slice_y_esima - centroid_y
            
            # Perform spline interpolation to smooth the slice
            x_spline, y_spline, z_spline = spline_interpolation(slice_x_trasl_esima, slice_y_trasl_esima, slice_z_esima)

            # Convert spline coordinates to pixel coordinates
            x_coord_pix2 = np.ceil((x_spline + centroid_x - image_position[0]) / pixel_spacing[0]).astype(int)
            y_coord_pix2 = np.ceil((y_spline + centroid_y - image_position[1]) / pixel_spacing[1]).astype(int)
            z_coord_pix2 = np.ceil((z_spline - image_position[2]) / z_spacing).astype(int)

            # Extract dose values from the dose matrix using the pixel coordinates
            dose_spline = dose_matrix[x_coord_pix2, y_coord_pix2, z_coord_pix2]
            
            # Check if the points are ordered clockwise or counterclockwise
            spline_coord = list(zip(x_spline, y_spline))
            bool_order = is_clockwise(spline_coord)

            # Reverse the spline points if they are counterclockwise
            if bool_order == True:
                x_spline_clock = x_spline[::-1]
                y_spline_clock = y_spline[::-1]
                dose_spline_clock = dose_spline[::-1]
            else:
                x_spline_clock = x_spline
                y_spline_clock = y_spline
                dose_spline_clock = dose_spline

            # Create a line representing the curve described by the points (x, y)
            slice_curve = LineString(zip(x_spline_clock, y_spline_clock))

            # Create a vertical line representing the y-axis (x=0)
            y_min, y_max = min(y_spline_clock), max(y_spline_clock)
            axis_y = LineString([(0, y_min), (0, y_max)])

            # Find the intersection between the curve and the y-axis
            intersection_y = slice_curve.intersection(axis_y)

            # Check and print the intersection point
            if intersection_y.is_empty:
                print("No intersection found.")
            else:
                print(f"Intersection point YY: {intersection_y}")
            
            # Extract y-coordinate values from intersection points
            y_value_coord = [point.y for point in intersection_y.geoms]
            
            # Determine point_cut based on the value of cut_open
            if cut_open == 'ant':
                point_cut = (0, min(y_value_coord))
            elif cut_open == 'post':
                point_cut = (0, max(y_value_coord))
            else:
                point_cut = (0, max(y_value_coord) if cut_open != 1 else min(y_value_coord))

            # Handle cases where there are multiple intersection points
            if len(intersection_y.geoms) == 4 and intersection_y.geom_type == 'MultiPoint':
                print('y values ', y_value_coord)
                y_value_coord_sort = sorted(y_value_coord)
                
                ansa_up = y_value_coord_sort[2]
                ansa_down = y_value_coord_sort[1]

                # Create a horizontal line to find x-axis intersections
                x_min, x_max = min(x_spline_clock), max(x_spline_clock)
                axis_x = LineString([(x_min, ansa_down), (x_max, ansa_down)])
                
                # Find intersection between the slice curve and the horizontal line
                intersection_x = slice_curve.intersection(axis_x)
                
                ref_x = min(intersection_x.geoms, key=lambda p: abs(p.x)).x
                
                # Check and print the intersection point on the x-axis
                if intersection_x.is_empty:
                    print("No intersection found.")
                else:
                    print(f"Intersection point XX: {intersection_x}")

                # Separate the intersection points based on their x-coordinates
                x_greater = [point.x for point in intersection_x.geoms if point.x > ref_x]
                x_lesser = [point.x for point in intersection_x.geoms if point.x < ref_x]
                
                # Filter out small values close to zero
                x_lesser_filtered = [x for x in x_lesser if abs(x) >= 0.05]
                x_greater_filtered = [x for x in x_greater if abs(x) >= 0.05]

                # Find the largest lesser and smallest greater values
                if x_lesser_filtered:
                    max_x_lesser = max(x_lesser_filtered)
                else:
                    max_x_lesser = None  # No lesser values found

                if x_greater_filtered:
                    min_x_greater = min(x_greater_filtered)
                else:
                    min_x_greater = None  # No greater values found
                    min_x_greater = 0.05  # Set a minimum threshold

                # Create points with corresponding doses
                points_clock_xy_dose = [((xi, yi), ti) for xi, yi, ti in zip(x_spline_clock, y_spline_clock, dose_spline_clock)]
                
                # Filter points based on the intersection boundaries
                points_ansa = [((xi, yi), ti) for (xi, yi), ti in points_clock_xy_dose if max_x_lesser <= xi <= min_x_greater and ansa_down <= yi <= ansa_up]
                                    
                tolerance = 1
                
                # Check if the first and last points are close to zero, indicating all points are included
                if (points_ansa[0][0][0] < tolerance or points_ansa[0][0][0] > -tolerance) and (points_ansa[-1][0][0] < tolerance or points_ansa[-1][0][0] > -tolerance):
                    print('ok')
                    points_ansa = points_ansa
                        
                # If the first point is near zero but the last isn't, filter until the nearest zero
                if (points_ansa[0][0][0] < tolerance or points_ansa[0][0][0] > -tolerance) and (points_ansa[-1][0][0] > tolerance or points_ansa[-1][0][0] < -tolerance):
                    # Find the first point with x near zero starting from the end of the list
                    point_near_zero = None
                    points_reversed_ansa = points_ansa[::-1]

                    for point in points_reversed_ansa:
                        x_value = point[0][0]
                        if abs(x_value) < tolerance:
                            point_near_zero = point
                            break
                            
                    if point_near_zero:
                        index_near_zero = len(points_ansa) - 1 - points_reversed_ansa.index(point_near_zero)
                    else:
                        print("No point found near zero.")
                        
                    points_ansa = points_ansa[0:index_near_zero + 1]

                # If the last point is near zero but the first isn't
                if (points_ansa[-1][0][0] < tolerance or points_ansa[-1][0][0] > -tolerance) and (points_ansa[0][0][0] > tolerance or points_ansa[0][0][0] < -tolerance):
                    # Find the first point with x near zero starting from the end of the list
                    point_near_zero = None
                    points_reversed_ansa = points_ansa

                    for point in points_reversed_ansa:
                        x_value = point[0][0]
                        if abs(x_value) < tolerance:
                            point_near_zero = point
                            break
                            
                    if point_near_zero:
                        index_near_zero = len(points_ansa) - 1 - points_reversed_ansa.index(point_near_zero)
                    else:
                        print("No point found near zero.")
                        
                    points_ansa = points_ansa[0:index_near_zero + 1]

                # Invert x-coordinates for the points within the ansa (loop)
                points_with_inverted_x = [((x * -1, y), t) for (x, y), t in points_ansa]
                
                # Update the original points with inverted x-coordinates
                points_clock_xy_dose_updated = [point if point not in points_ansa else points_with_inverted_x[points_ansa.index(point)] for point in points_clock_xy_dose]

                x_spline_clock = [point[0][0] for point in points_clock_xy_dose_updated]
                y_spline_clock = [point[0][1] for point in points_clock_xy_dose_updated]
                dose_spline_clock = [point[1] for point in points_clock_xy_dose_updated]
            
            # Check if the intersection yields multiple cross points
            if len(intersection_y.geoms) != 2:
                print('multi cross')

            # Separate positive and negative values for dose analysis
            x_pos, y_pos, dose_pos, x_neg, y_neg, dose_neg = separate_positive_negative_values(x_spline_clock, y_spline_clock, dose_spline_clock)

            # Find the nearest points to the cut point for both positive and negative values
            point_pos_start = find_nearest_point(point_cut[0], point_cut[1], x_pos, y_pos)
            point_neg_start = find_nearest_point(point_cut[0], point_cut[1], x_neg, y_neg)

            points_pos = list(zip(x_pos, y_pos))
            points_neg = list(zip(x_neg, y_neg))

            # Sort the positive points by distance to the starting point
            xy_points_pos_sort = sort_points_by_greedy_distance(points_pos, point_pos_start)
            dose_pos_sort = reorder_dose(dose_pos, points_pos, xy_points_pos_sort)

            # Sort the negative points by distance to the starting point
            xy_points_neg_sort = sort_points_by_greedy_distance(points_neg, point_neg_start)
            dose_neg_sort = reorder_dose(dose_neg, points_neg, xy_points_neg_sort)

            #### Unfolding points ####
            xy_points_pos_sort = xy_points_pos_sort[::-1]  # Reverse order for anterior cuts
            distance_points_pos = cumulative_distances(xy_points_pos_sort)
            
            mean_doses_pos = []
            # Calculate mean doses between sorted positive points
            for a in range(len(xy_points_pos_sort) - 1):
                mean_doses_pos_tmp = np.mean([dose_pos_sort[a + 1], dose_pos_sort[a]])
                mean_doses_pos.append(mean_doses_pos_tmp)
            
            mean_doses_pos = mean_doses_pos[::-1]  # Reverse for anterior cuts
            z_slice_unfolding_pos = np.full(len(distance_points_pos), slice_z_esima[0])

            # Process negative points similarly
            distance_points_neg = cumulative_distances(xy_points_neg_sort)

            mean_doses_neg = []
            # Calculate mean doses between sorted negative points
            for a in range(len(xy_points_neg_sort) - 1):
                mean_doses_neg_tmp = np.mean([dose_neg_sort[a + 1], dose_neg_sort[a]])
                mean_doses_neg.append(mean_doses_neg_tmp)
            
            z_slice_unfolding_neg = np.full(len(distance_points_neg), slice_z_esima[0])

            # Reverse the distance for negative points
            distance_points_neg = (distance_points_neg[::-1])
            flip_points = -1
            
            distance_points_rev_neg = [x * flip_points for x in distance_points_neg]

            # Combine unfolded x, z, and dose values
            unfolding_esima_x = np.hstack((distance_points_rev_neg, distance_points_pos))
            unfolding_esima_z = np.hstack((z_slice_unfolding_neg, z_slice_unfolding_pos))
            unfolding_esima_dose = np.hstack((mean_doses_neg, mean_doses_pos))
            
            # Append unfolding data for the current slice
            unfolding_slice_x.append(unfolding_esima_x)
            unfolding_slice_z.append(unfolding_esima_z)
            unfolding_slice_dose.append(unfolding_esima_dose)

    # Plot the unfolded slices
    fig_point = plt.scatter(unfolding_slice_x, unfolding_slice_z, c=unfolding_slice_dose, cmap='turbo')
    plt.colorbar(label='Dose [Gy]')
    plt.axis('image')
    plt.xlabel('X [mm]')
    plt.ylabel('Z [mm]')
    plt.close()
    plt.clf()
    
    unfolding_slice_x = np.asarray(unfolding_slice_x)
    unfolding_slice_z = np.asarray(unfolding_slice_z)
    unfolding_slice_dose = np.asarray(unfolding_slice_dose)
    
    return fig_point, unfolding_slice_x, unfolding_slice_z, unfolding_slice_dose
