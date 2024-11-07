#!/usr/bin/env python

"""utilities.py: Utility functions for DSM/DVH/DSH processing."""

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os


def find_file_with_string(search_string, file_list):
    """Searches for a file in a list containing a specific string."""
    search_string = search_string.lower()# Convert the search string to lowercase
    for file_path in file_list:
        if search_string in file_path.lower():
            return file_path
    return None

def check_consecutive(numbers):
    """Checks if the given list of numbers is consecutive."""
    # Convert the list of strings to floating-point numbers
    numbers = list(map(float, numbers))
    numbers.sort()
    
    if len(numbers) < 2:
        return True
    
    # Calculate the step between the first two numbers
    step = numbers[1] - numbers[0]
    # Check if the step is the same for all consecutive numbers
    for i in range(1, len(numbers) - 1):
        if numbers[i + 1] - numbers[i] != step:
            return False
    
    return True


def quantify_concavity(z_spacing_slice, z_conc):
    """Quantifies the concavity of the structure based on z coordinates."""
    # Convert lists of strings to float
    z_slice = list(map(float, z_spacing_slice))
    z_concavity = list(map(float, z_conc))
    z_slice = sorted(z_slice)
    # Calculate the structural height
    height_struct = z_slice[-1] - z_slice[0]
    # Calculate the structural height divided by 3
    height_struct_thirds = height_struct // 3
    
    # Check the position of z_concavity within z_slice
    def is_sublist(whole_list, sub_list):
        sub_list_len = len(sub_list)
        for i in range(len(whole_list) - sub_list_len + 1):
            if whole_list[i:i + sub_list_len] == sub_list:
                return True, i
        return False, -1
    
    found, position_index = is_sublist(z_slice, z_concavity)
    
    if found:
        start_diff = z_concavity[0] - z_slice[0]
        end_diff = z_slice[-1] - z_concavity[-1]
        
        # Check if the distance between z_concavity and z_slice is greater than height_struct_thirds
        if start_diff > height_struct_thirds or end_diff > height_struct_thirds:
            result_list = [item for item in z_slice if float(item) not in z_concavity]
            
            # Determine the position of z_concavity relative to z_spacing_slice
            if position_index == 0:
                position = "down"
            elif position_index + len(z_concavity) == len(z_slice):
                position = "up"
            else:
                position = "middle"
            
            print(f"z_concavity is located {position} of z_spacing_slice.")
            return result_list, position
        else:
            print("The distance between z_concavity and z_spacing_slice is not greater than height_struct_thirds.")
            return [], None
    else:
        print("z_concavity is not found within z_spacing_slice.")
        return [], None


def check_concavity(info_struct):
    """Checks the concavity of the given structure information."""
    data = dict(info_struct)
    z_spacing_slice, z_concavity = [], []
    for key, value in info_struct:
        z_spacing_slice.append(key)
        if len(data[key]) > 1:
            z_concavity.append(key)
        
    if (check_consecutive(z_spacing_slice) == True) and len(z_concavity) == 0:
        return True
        
    if (check_consecutive(z_spacing_slice) == True) and len(z_concavity) > 0:
        part_to_extract_map, pos = quantify_concavity(z_spacing_slice, z_concavity)
        return part_to_extract_map, pos
    if (check_consecutive(z_spacing_slice) == False):
        print(z_spacing_slice)
        return False


def save_img(plot, path_to_save, patient_name, structure, name_plot, cut, matrix):
    """Saves a plot and associated data to specified paths."""
    if matrix is not None:
        plot.figure.savefig(os.path.join(path_to_save, f'{patient_name}_{structure}{name_plot}{cut}.png'))
        np.save(os.path.join(path_to_save, f'{patient_name}_{structure}{name_plot}{cut}.npy'), matrix)
    else:
        plot.figure.savefig(os.path.join(path_to_save, f'{patient_name}_{structure}{name_plot}{cut}.png'))


def spline_interpolation(slice_x_translation, slice_y_translation, slice_z):
    """Performs spline interpolation on given x, y coordinates at a specific z level."""
    try:
        tck, u = splprep([slice_x_translation, slice_y_translation], u=None, s=0.0, per=1)
        smooth_spline = np.linspace(u.min(), u.max(), 1000)
        x_spline, y_spline = splev(smooth_spline, tck, der=0)
        z_spline = np.empty(1000)
        z_spline.fill(slice_z[0])
        
        return x_spline, y_spline, z_spline
    
    except ValueError as e:
        return fallback_function(slice_x_translation, slice_y_translation, slice_z)


def is_clockwise(poly):
    """Determines if the polygon defined by poly is oriented clockwise."""
    total = poly[-1][0] * poly[0][1] - poly[0][0] * poly[-1][1]
    for i in range(len(poly) - 1):
        total += poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]

    return total <= 0


def separate_positive_negative_values(x, y, dose):
    """Separates the x, y, and dose values into positive and negative based on x values."""
    x_pos, y_pos, dose_pos = [], [], []
    x_neg, y_neg, dose_neg = [], [], []
    for i in range(len(x)):
        if x[i] >= 0:
            x_pos.append(x[i])
            y_pos.append(y[i])
            dose_pos.append(dose[i])
        else:
            x_neg.append(x[i])
            y_neg.append(y[i])
            dose_neg.append(dose[i])

    return x_pos, y_pos, dose_pos, x_neg, y_neg, dose_neg
 
 
def find_nearest_point(target_x, target_y, points_x, points_y):
    """Finds the nearest point in a set of points to a given target point."""
    # Convert input lists to numpy arrays for vectorized operations
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    
    # Compute the squared Euclidean distance between the target point and each point in the vector
    distances = (points_x - target_x)**2 + (points_y - target_y)**2
    
    # Find the index of the minimum distance
    nearest_index = np.argmin(distances)
    nearest_point = (points_x[nearest_index], points_y[nearest_index])
    
    return nearest_point


def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def sort_points_by_greedy_distance(points, start_point):
    """Sorts points based on the greedy nearest neighbor algorithm starting from a given point."""
    # Initialize the sorted points list with the starting point
    sorted_points = [start_point]
    # Create a list of points to visit (excluding the starting point)
    points_to_visit = points.copy()
    
    current_point = start_point
    while points_to_visit:
        # Find the closest point to the current point
        distances = [euclidean_distance(current_point, point) for point in points_to_visit]
        nearest_index = np.argmin(distances)
        nearest_point = points_to_visit[nearest_index]
        # Add the found point to the sorted points list
        sorted_points.append(nearest_point)
        # Update the current point
        current_point = nearest_point
        # Remove the found point from the list of points to visit
        points_to_visit.pop(nearest_index)
    
    return sorted_points


def reorder_dose(dose, points, sorted_points):
    """Reorders the dose vector according to the sorted points."""
    # Create a dictionary to find the original index of each point
    point_to_index = {point: index for index, point in enumerate(points)}
    sorted_indices = [point_to_index[point] for point in sorted_points]
    # Reorder the dose vector using the sorted indices
    dose_sorted = [dose[index] for index in sorted_indices]
    return dose_sorted


def cumulative_distances(points):
    """Calculates the cumulative distances between consecutive points."""
    distances = []
    cumulative_distance = 0.0
    for i in range(len(points) - 1):
        distance = euclidean_distance(points[i], points[i + 1])
        cumulative_distance += distance
        distances.append(cumulative_distance)
    return distances

