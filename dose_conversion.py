#!/usr/bin/env python

"""
dose_conversion.py: This module contains functions for converting dose matrices to EQD2 based on prescribed dose information extracted from an .xlsx file.
"""

import os
import sys
import getopt
import pydicom as py
import pandas as pd
from pydicom.data import get_testdata_file
from pydicom.uid import ImplicitVRLittleEndian
import numpy as np

def convert_eqd2(dcm_dose, list_param):

    dcm_dose = py.read_file(dcm_dose, force=True)
    name_patient = dcm_dose.PatientID

    # Parse the parameters from the list_param string
    param = list_param.split(',')
    file_dose_info = param[0]  # Dose information Excel file name
    alfa_beta = float(param[1])  # Alpha/Beta ratio
    gamma_time = float(param[2])  # Timing correction factor

    # Load dose information from the specified Excel file
    df_info_dose = pd.read_excel(file_dose_info, index_col=None, header=None)
    info_dose_file_current_pts = df_info_dose.loc[df_info_dose[0] == name_patient]

    # Check if patient information is found
    if info_dose_file_current_pts.shape[0] > 0:
        dose_prt = info_dose_file_current_pts[1].tolist()[0]  # Prescribed dose
        number_fract = info_dose_file_current_pts[2].tolist()[0]  # Number of fractions
        days_RT = info_dose_file_current_pts[3].tolist()[0]  # Days for RT

        number_bit = dcm_dose.BitsStored  # Determine the bit depth of the dose data
        if number_bit == 16:
            arr = dcm_dose.pixel_array
            # Convert pixel data to 32-bit for processing
            dcm_dose.PixelData = arr.astype('int32').tobytes()
            dcm_dose.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            dcm_dose.is_implicit_VR = True
            dcm_dose['PixelData'].VR = 'OB'
            dcm_dose.BitsAllocated = 32
            dcm_dose.BitsStored = 32
            dcm_dose.HighBit = 31  # Adjust for 32-bit depth
            matrix_to_transform = arr * dcm_dose.DoseGridScaling
        elif number_bit == 32:
            print("32-bit data detected.")
            matrix_to_transform = dcm_dose.pixel_array * dcm_dose.DoseGridScaling
            
        # Calculate EQD2 using the prescribed dose and fractionation parameters
        Gy_fract = dose_prt / number_fract  # Dose per fraction
        d_esimo_eq2_tmp = matrix_to_transform * ((matrix_to_transform / number_fract + alfa_beta) / (alfa_beta + 2))
        T2Gy = ((dose_prt * (Gy_fract + alfa_beta) / (2 + alfa_beta)) / 2) / 5 * 7  # Time to EQD2
        dos_corretta = dose_prt * ((dose_prt / number_fract + alfa_beta) / (alfa_beta + 2))
        balance_time = matrix_to_transform / dos_corretta  # Balance time calculation
        time_parm = gamma_time * balance_time * (T2Gy - days_RT)  # Time adjustment
        D_matrix_eq2_tmp = d_esimo_eq2_tmp + time_parm  # Final adjusted dose matrix

        print("Max value in EQD2 matrix:", D_matrix_eq2_tmp.max())
        matrix_pixel = np.transpose(D_matrix_eq2_tmp)  
        return matrix_pixel
        
    else:
        print('Patient', name_patient, 'not found in the list.')
        return None

