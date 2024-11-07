#!/usr/bin/env python

"""
mode_function.py: This module contains functions for selecting and calculating dose metrics based on specified modes. It handles dose metrics DSM (calculations for both anterior and posterior orientations), DVH and DSH. The results are logged to a specified CSV file for further analysis.
"""

import os
import sys
import getopt
import pydicom as py
from dsm_algorithm import calculate_dsm
from dvh_algorithm import calculate_dvh
from dsh_algorithm import calculate_dsh
import pandas as pd

import csv


def select_metric_dsm(arg_mode, arg_str, dose_matrix, file_struct, file_dose, norm_param, metric_info, start_time):

    # Open the CSV file to log patient information issues
    with open(metric_info, 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)

        # Calculate DSM for anterior orientation if mode is DSM-a
        if arg_mode == 'DSM-a' and arg_str != '':
            print(f"Calculating DSM-a with norm_param: {norm_param}")
            cut_open = 'ant'
            id, type_problem = calculate_dsm(file_struct, file_dose, dose_matrix, arg_str, cut_open, norm_param, start_time)
            csv_writer.writerow([id, type_problem])

        # Calculate DSM for posterior orientation if mode is DSM-p
        if arg_mode == 'DSM-p' and arg_str != '':
            print(f"Calculating DSM-p with norm_param: {norm_param}")
            cut_open = 'post'
            id, type_problem = calculate_dsm(file_struct, file_dose, dose_matrix, arg_str, cut_open, norm_param, start_time)
            csv_writer.writerow([id, type_problem])


def select_metric_dvh(arg_mode, arg_str, dose_matrix, file_struct, file_dose, metric_info, bin_param):

    # Open the CSV file to log patient information issues
    with open(metric_info, 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)

        # Calculate DSM for anterior orientation if mode is DSM-a
        if arg_mode == 'DVH' and arg_str != '':
            dvh_data = calculate_dvh(file_struct, file_dose, arg_str, bin_param)
            dvh_data.to_csv(metric_info, mode='a', header=False, index=False)


def select_metric_dsh(arg_mode, arg_str, dose_matrix, file_struct, file_dose, metric_info, file_stl, bin_param):

    # Open the CSV file to log patient information issues
    with open(metric_info, 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)

        # Calculate DSM for anterior orientation if mode is DSM-a
        if arg_mode == 'DSH' and arg_str != '':
            dsh_data = calculate_dsh(file_dose, arg_str, file_stl, bin_param)
            print('tipoo ', type(dsh_data))
            if isinstance(dsh_data, pd.DataFrame) and dsh_data is not None:
                dsh_data.to_csv(metric_info, mode='a', header=False, index=False)
