# Tumor Segmentation Predictions Using RH-GlioSeg-nnUNet with Preprocessed NifTI Files
# ==============================================================================#
#  Author:       * Roberto Romero-Oraá and + Santiago Cepeda                    #
#  Copyright:    * Biomedical Engineering Group                                #
#                + Río Hortega University Hospital                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# ==============================================================================#

# This script is tailored for processing single-timepoint scans. It expects input files that have already undergone coregistration and skull stripping.

import os
import shutil
import argparse
import subprocess
import nibabel as nib
import time
import utils
import registration
import segmentation

# Parse command line arguments
parser = argparse.ArgumentParser(description="NifTI image processing and tumor subregion segmentation")
parser.add_argument("-i", "--input", required=True, help="Input directory with subject subfolders containing processed NifTI files")
parser.add_argument("-o", "--output", required=True, help="Output directory for the segmentation results.")
args = parser.parse_args()

# Path to the directory where the script is located
dir_path = os.path.dirname(os.path.realpath(__file__))
atlas_image = os.path.join(dir_path, 'atlas_sri24', 'atlasImage_sk.nii.gz')

# Process each subject in the input directory
for subject_id in os.listdir(args.input):
    subject_path = os.path.join(args.input, subject_id)
    if not os.path.isdir(subject_path):
        continue
    print(f"[Subject {subject_id}]")
    start = time.time()

    # Output directory for this subject
    subject_output_path = os.path.join(args.output, subject_id)
    os.makedirs(subject_output_path, exist_ok=True)

    # Process MR sequences: Registration and Z-score normalization
    for seq in ['t1', 't1ce', 't2', 'flair']:
        input_seq_path = os.path.join(subject_path, f"{seq}.nii.gz")
        if os.path.exists(input_seq_path):
            print(f"Processing {seq} sequence... ", end='')
            # Registration
            registered_seq_path = os.path.join(subject_output_path, f"{seq}_reg.nii.gz")
            registration.run(in_file=input_seq_path, reference=atlas_image, out_file=registered_seq_path)
            # Z-score normalization
            normalized_seq_path = os.path.join(subject_output_path, f"{seq}_norm.nii.gz")
            utils.normalize_intensity(registered_seq_path, normalized_seq_path)
            print("✓")

    # Segmentation with nnUNet
    print("Running nnUNet segmentation... ", end='')
    nnunet_input_folder = os.path.join(subject_output_path, "nnUNet_input")
    nnunet_output_folder = os.path.join(subject_output_path, "nnUNet_output")
    os.makedirs(nnunet_input_folder, exist_ok=True)
    os.makedirs(nnunet_output_folder, exist_ok=True)

    # Prepare files for nnUNet
    file_mapping = {
        'flair_norm.nii.gz': f'{subject_id}_0000.nii.gz',
        't1_norm.nii.gz': f'{subject_id}_0001.nii.gz',
        't1ce_norm.nii.gz': f'{subject_id}_0002.nii.gz',
        't2_norm.nii.gz': f'{subject_id}_0003.nii.gz',
    }
    for original, renamed in file_mapping.items():
        shutil.copy(os.path.join(subject_output_path, original), os.path.join(nnunet_input_folder, renamed))

    # Execute nnUNet prediction
    segmentation.run(nnunet_input_folder, nnunet_output_folder)
    print("✓")

    # Post-processing and cleanup
    # Rename and remove unnecessary files
    for filename in os.listdir(subject_output_path):
        file_path = os.path.join(subject_output_path, filename)
        if filename.endswith('_norm.nii.gz'):
            new_filename = filename.replace('_norm', '')
            new_file_path = os.path.join(subject_output_path, new_filename)
            shutil.move(file_path, new_file_path)
        elif filename.endswith('_reg.nii.gz'):
            os.remove(file_path)

    # Move the segmentation file, if it exists
    expected_segmentation_file = f'{subject_id}.nii.gz'
    segmentation_file_path = os.path.join(nnunet_output_folder, expected_segmentation_file)
    if os.path.exists(segmentation_file_path):
        final_segmentation_path = os.path.join(subject_output_path, "segmentation.nii.gz")
        shutil.move(segmentation_file_path, final_segmentation_path)
        print(f"Segmentation file saved to {final_segmentation_path}")
    else:
        print("Error: Expected segmentation file not found.")

    shutil.rmtree(nnunet_input_folder)
    shutil.rmtree(nnunet_output_folder)
    print(f"Completed processing for subject {subject_id}.")

    end = time.time()
    print(time.strftime('Subject processing time: %H:%M:%S', time.gmtime(end - start)))

print("All processing completed.")
