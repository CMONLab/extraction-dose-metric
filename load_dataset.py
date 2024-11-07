#!/usr/bin/env python

"""load_dataset.py:Script to process DICOM RTDOSE and RTSTRUCT files within a specified directory. This script scans through folders, identifying relevant DICOM files for specific dose metrics extraction and dose conversion if specified. It handles cases where more than one dose or structure file is present and logs directories with missing or conflicting files. """

import os
import sys
import getopt
import pydicom as py
from dose_conversion import convert_eqd2
from mode_function import select_metric_dsm, select_metric_dvh, select_metric_dsh
import csv

# Main function to process DICOM files in the specified directory
def load_db(path_db, arg_str, arg_eqd2, arg_mode, arg_extra_param, pts_no_info, pts_info_issue, start_time):
    
    # Prepare CSV writer for logging directories with missing information
    with open(pts_no_info, 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)

        # Iterate through directories and files in the given path
        for path, dirs, files in os.walk(path_db):
            if path != path_db:
                print("Current directory:", path)
                file_doses, file_structures, file_slt = [], [], []
                # Identify RTDOSE and RTSTRUCT files
                for f in files:
                    if not f.startswith('.') and os.path.isfile(os.path.join(path, f)):
                        name_file = os.path.join(path, f)
                        file, extension = os.path.splitext(f)
                        data_dcm = py.read_file(name_file, force=True)

                        if extension == '.dcm' and data_dcm.Modality == "RTDOSE":
                            file_doses.append(name_file)
                        elif extension == '.dcm' and data_dcm.Modality == "RTSTRUCT":
                            file_structures.append(name_file)
                        elif extension == '.stl':
                            file_slt.append(name_file)


                # Handling multiple or missing structure files
                if len(file_structures) == 1:
                    file_struct_to_use = file_structures[0]
                    print("One structure file found.")
                else:
                    print("Zero or multiple structure files found.")
                    file_struct_to_use = None

                # Handling multiple or missing dose files
                if len(file_doses) == 1:
                    file_dose_to_use = file_doses[0]
                    print("One dose file found.")
                elif len(file_doses) > 1:
                    file_dose_to_use = None
                    for file_dose in file_doses:
                        if os.path.basename(file_dose) == "dose_tot.dcm":
                            file_dose_to_use = file_dose
                            break
                    if file_dose_to_use is None:
                        print("Multiple dose files found, but none named 'dose_tot.dcm'.")
                else:
                    print("No dose file found.")
                    file_dose_to_use = None


                # Check for the presence of necessary files
                if file_struct_to_use and file_dose_to_use:
                    # Set default values for arg_extra_param
                    if arg_mode not in ['DVH', 'DSH'] and not arg_extra_param:
                        arg_extra_param = '100,50,0'
                    elif arg_mode in ['DVH', 'DSH'] and not arg_extra_param:
                        arg_extra_param = '100'

                    if arg_eqd2:
                        ## Convert dose in base on arg_eqd2
                        print("Applying EQD2 conversion")
                        dose_matrix = convert_eqd2(file_dose_to_use, arg_eqd2)
                        if dose_matrix is not None:

                            # Selection of the appropriate function based on arg_mode
                            if arg_mode in ['DSM-a', 'DSM-p']:
                                select_metric_dsm(arg_mode, arg_str, dose_matrix,
                                                  file_struct_to_use, file_dose_to_use,
                                                  arg_extra_param, pts_info_issue, start_time)
                            elif arg_mode == 'DVH':
                                select_metric_dvh(arg_mode, arg_str, dose_matrix,
                                                  file_struct_to_use, file_dose_to_use,
                                                  pts_info_issue, arg_extra_param)
                            elif arg_mode == 'DSH':
                                select_metric_dsh(arg_mode, arg_str, dose_matrix,
                                                  file_struct_to_use, file_dose_to_use,
                                                  pts_info_issue, file_slt, arg_extra_param)
                        else:
                            # Log missing dose information
                            csv_writer.writerow([os.path.basename(os.path.normpath(path))])
                    
                    else:
                        ## Use orginal dose 
                        print("No EQD2 conversion applied")
                        dose_matrix = py.read_file(file_dose_to_use, force=True)
                        if arg_mode in ['DSM-a', 'DSM-p']:
                            select_metric_dsm(arg_mode, arg_str, dose_matrix,file_struct_to_use, file_dose_to_use, arg_extra_param, pts_info_issue, start_time)
                        elif arg_mode == 'DVH':
                            select_metric_dvh(arg_mode, arg_str, dose_matrix, file_struct_to_use, file_dose_to_use, pts_info_issue, arg_extra_param)
                        elif arg_mode == 'DSH':
                            select_metric_dsh(arg_mode, arg_str, dose_matrix, file_struct_to_use, file_dose_to_use, pts_info_issue, file_slt, arg_extra_param)


                        
                else:
                    print("Required files not found or multiple conflicting files present.")

