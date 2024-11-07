#!/usr/bin/env python

"""main.py:Script to extract dose metrics (DVH, DSH, DSM) for specific structures from DICOM datasets."""

import os
import sys
import getopt
from load_dataset import load_db
import csv
import time
from datetime import datetime



def main(argv):
    # Initialize input arguments with default empty values
    arg_in = ""
    arg_out = ""
    arg_mode = ""
    arg_str = ""
    arg_eqd2 = ""
    arg_extra_param = ""

    # Help message with usage example
    arg_help = (f"{argv[0]} -i <input:db directory> -o <output:output directory> "
                "-m <mode: metric to extract [DVH/DSH/DSM-a/DSM-p]> -s <structure [list]> "
                "-c <convert the dose to EQD2 [treatment_details_file, a/b, y]> "
                "-x <normalization for DSM-a/DSM-p; [Nx,Ny,15/30/0] or binning for DVH/DSH [default 100 for 1 Gy]>")

    # Parse command-line arguments
    try:
        opts, args = getopt.getopt(argv[1:], "h:i:o:m:s:c:x:",
                                   ["help", "input_directory=", "output_directory=",
                                    "mode=", "structure=", "eqd2_convert=", "extra_param="])
    except getopt.GetoptError:
        print(arg_help)
        sys.exit(2)

    # Map arguments to variables
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit()
        elif opt in ("-i", "--input_directory"):
            arg_in = arg
        elif opt in ("-o", "--output_directory"):
            arg_out = arg
        elif opt in ("-m", "--mode"):
            arg_mode = arg
        elif opt in ("-s", "--structure"):
            arg_str = arg
        elif opt in ("-c", "--eqd2_convert"):
            arg_eqd2 = arg
        elif opt in ("-x", "--extra_param"):
            arg_extra_param = arg

    # Display parsed arguments
    print(f'Input directory: {arg_in}')
    print(f'Output directory: {arg_out}')
    print(f'Metric to extract: {arg_mode}')
    print(f'Structure: {arg_str}')
    print(f'Conversion to EQD2: {arg_eqd2}')
    print(f'Extra parameter for normalization/binning: {arg_extra_param}')

    # Validate required arguments
    if not arg_str:
        print("Please specify a structure or list of structures (-s 'structure')")
        sys.exit(2)
    if not arg_in:
        print("Please specify the input directory (-i 'dir_path')")
        sys.exit(2)
    if not arg_out:
        print("Please specify the output directory (-o 'dir_path')")
        sys.exit(2)
    if not arg_mode:
        print("Please specify the mode (-m 'metric to extract')")
        sys.exit(2)
        
    # Prepare current date for file naming
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Define file names based on date and metric
    pts_no_info = os.path.join(arg_out, f'pts_no_info_{current_date}.csv')
    
    # Initialize pts_no_info if EQD2 conversion is specified
    if arg_eqd2:
        with open(pts_no_info, 'w', newline='') as out_info:
            csv_writer = csv.writer(out_info)
            csv_writer.writerow(['Patient ID'])  # Header for missing patient info

    # Initialize output CSV file with headers based on arg_mode
    if arg_mode.startswith("DSM"):
        pts_info_out = os.path.join(arg_out, f'pts_info_concavity_{arg_str}_{current_date}.csv')
        with open(pts_info_out, 'w', newline='') as out_issue:
            csv_writer = csv.writer(out_issue)
            csv_writer.writerow(['Patient ID', 'Concavity Issue'])
    elif arg_mode == "DVH":
        pts_info_out = os.path.join(arg_out, f'DVH_output_{arg_str}_{current_date}.csv')
        with open(pts_info_out, 'w', newline='') as out_dvh:
            csv_writer = csv.writer(out_dvh)
            csv_writer.writerow(['Patient ID', 'structure_name', 'Volume [cm3]', 'Max_dose_plan [Gy]', 'Max_dose_str [Gy]', 'Min_dose_str [Gy]', 'Mean_dose_str [GY] ', '0'])
    elif arg_mode == "DSH":
        pts_info_out = os.path.join(arg_out, f'DSH_output_{arg_str}_{current_date}.csv')
        with open(pts_info_out, 'w', newline='') as out_dsh:
            csv_writer = csv.writer(out_dsh)
            csv_writer.writerow(['Patient ID', 'structure_name' ,'Surface Tot [mm2]', 'maxDose', '0'])

    # Record start time for operation timing
    start_time = time.time()

    # Load the database and process metric extraction
    load_db(arg_in, arg_str, arg_eqd2, arg_mode, arg_extra_param, pts_no_info, pts_info_out, start_time)



if __name__ == "__main__":
    main(sys.argv)

