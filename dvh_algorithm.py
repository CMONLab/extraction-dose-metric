#!/usr/bin/env python

"""dvh_algorithm.py: Calculate Dose Volume Histogram (DVH) for specified structures. """

import pydicom as py
import os
import sys
import re
from dicompylercore import dicomparser, dvhcalc
import pandas as pd
import numpy as np
from typing import List, Dict, Union


def calculate_dvh(rtssfile: str, rtdosefile: str, structure_name: str, bin: int) -> pd.DataFrame:

    """
    Args:
        rtssfile: Path to the RT Structure Set DICOM file.
        rtdosefile: Path to the RT Dose DICOM file.
        structure_name: Comma-separated string of structure names to include.
        bin: Number of bins for the DVH calculation.

    Returns:
        A DataFrame containing DVH data and associated information.
    """
    
    # Load the dose data
    data_dose = py.read_file(rtdosefile, force=True)
    
    # Extract patient ID and initialize structure parser
    name_patient = data_dose.PatientID
    RTss = dicomparser.DicomParser(rtssfile)
    RTstructures = RTss.GetStructures()
    
    # Calculate the maximum dose value
    max_value_dose = np.max(data_dose.pixel_array * data_dose.DoseGridScaling)

    # Split the structure names into a list
    structure_names = structure_name.split(',')

    calcdvhs = {}
    data_info, data_dvh = [], []
    for key, structure in RTstructures.items():
        for s in structure_names:
            if re.match(s.strip(), structure['name'], re.IGNORECASE):  # Strip whitespace from structure names
                all_info_slice_structure = RTss.GetStructureCoordinates(key).items()
                keys_all_info_slice = [key for key in all_info_slice_structure]
                number_slice_structure = len(keys_all_info_slice)

                if number_slice_structure > 1:
                    calcdvhs[key] = dvhcalc.get_dvh(rtssfile, rtdosefile, key, None, False)
                    if (key in calcdvhs) and (len(calcdvhs[key].counts) and calcdvhs[key].counts[0] != 0):
                        print(f'DVH found for {structure["name"]}')
                        # Calculate the histogram values and round them
                        value_histogram = np.round(calcdvhs[key].counts[::int(bin)], 2)
                        dmax = np.round(calcdvhs[key].max, 2)
                        dmin = np.round(calcdvhs[key].min, 2)
                        dmean = np.round(calcdvhs[key].mean, 2)

                        data_info.append([
                            name_patient,
                            structure['name'],
                            np.round(calcdvhs[key].volume, 2),
                            np.round(max_value_dose, 2),
                            dmax,
                            dmin,
                            dmean
                        ])
                        data_dvh.append(value_histogram)


    data_info_save = pd.DataFrame(data_info)
    data_dvh_save = pd.DataFrame(data_dvh)
    data_to_save = pd.concat([data_info_save, data_dvh_save], axis=1)

    return data_to_save


