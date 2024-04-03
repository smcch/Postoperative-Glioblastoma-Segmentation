# ==============================================================================#
#  Author:       * Roberto Romero-Oraá and + Santiago Cepeda                        #
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

import os
import glob
import shutil
import argparse
import dcm2niix

import utils
import computeADC
import mri_synthstrip
import simpleElastix
from rhuh_nnunet_predict import run_nnunet_predict

# Parse command line arguments
parser = argparse.ArgumentParser(description="Processing DICOM images and segmentation of tumor subregions")
parser.add_argument("-i", "--input", required=True, help="Input directory with the DICOM files.")
parser.add_argument("-o", "--output", required=True, help="Output directory for processing results.")
args = parser.parse_args()

# Obtain the path to the directory where main.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

# Utilize the passed arguments to configure the input and output directories
path_dicom = args.input
path_nifti = args.output

# Atlas image directory
atlas_image = os.path.join(dir_path, 'atlas_sri24', 'atlastImage.nii.gz')

# Iterate every subject in the input directory
for subject_id in os.listdir(path_dicom):
    subject_path = os.path.join(path_dicom, subject_id)
    if not os.path.isdir(subject_path):
        continue

    # Get time points from subfolders
    time_points = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

    # Iterate time points
    for time_point in time_points:
        time_point_path = os.path.join(subject_path, time_point)
        output_path = os.path.join(path_nifti, subject_id, time_point)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compute 'dwi' or 'adc'.
        for seq in ['dwi', 'adc']:
            seq_path = os.path.join(time_point_path, seq)
            if os.path.exists(seq_path):
                os.system(f"{dcm2niix.bin} -z y -m n -b n -o {output_path} -f {seq} {seq_path}")

                if seq == 'dwi':
                    # Compute 'adc' fom 'dwi'
                    computeADC.run(
                        dwiPath=os.path.join(output_path, 'dwi.nii.gz'),
                        bvalPath=os.path.join(output_path, 'dwi.bval'),
                        adcPath=os.path.join(output_path, 'adc.nii.gz')
                    )
                break

        # Process other anatomical MR sequences
        for mri_seq in ['flair', 't1', 't1ce', 't2']:
            seq_path = os.path.join(time_point_path, mri_seq)
            if os.path.exists(seq_path):
                os.system(f"{dcm2niix.bin} -z y -m n -b n -o {output_path} -f {mri_seq} {seq_path}")

        # Verify and correct filenames
        utils.remove_unwanted_suffixes(output_path)

        # Registration, skull stripping, and intensity normalization for all sequences except 'adc'
        sequences_to_process = ['t1ce', 't1', 't2', 'flair']
        for mri_seq in sequences_to_process:
            in_file = os.path.join(output_path, f"{mri_seq}.nii.gz")
            if os.path.exists(in_file):
                # Registration
                reference_image = atlas_image if mri_seq == 't1ce' else os.path.join(output_path, 't1ce_reg.nii.gz')
                out_file = os.path.join(output_path, f"{mri_seq}_reg.nii.gz")
                simpleElastix.run(in_file=in_file, reference=reference_image, out_file=out_file)

                # Skull stripping
                skull_stripped_file = os.path.join(output_path, f"{mri_seq}_reg_sk.nii.gz")
                mri_synthstrip.run(
                    image=out_file,
                    out=skull_stripped_file,
                    mask=os.path.join(output_path, f"{mri_seq}_mask.nii.gz"),
                    modelPath=os.path.join(dir_path, 'synthstrip_models')
                )

                # Intensity normalization
                normalized_file = os.path.join(output_path, f"{mri_seq}_norm.nii.gz")
                utils.normalize_intensity(
                    image_path=skull_stripped_file,
                    output_path=normalized_file
                )

        # Process adc.nii.gz after t1ce_reg_sk.nii.gz
        if os.path.exists(os.path.join(output_path, 'adc.nii.gz')):
            # Skull Stripping for adc
            adc_skull_stripped_file = os.path.join(output_path, 'adc_sk.nii.gz')
            mri_synthstrip.run(
                image=os.path.join(output_path, 'adc.nii.gz'),
                out=adc_skull_stripped_file,
                mask=os.path.join(output_path, 'adc_mask.nii.gz'),
                modelPath=os.path.join(dir_path, 'synthstrip_models')
            )

            # Registration for adc (using t1ce_reg_sk.nii.gz)
            adc_registered_file = os.path.join(output_path, 'adc_reg.nii.gz')
            simpleElastix.run(
                in_file=adc_skull_stripped_file,
                reference=os.path.join(output_path, 't1ce_reg_sk.nii.gz'),
                out_file=adc_registered_file
            )

            # Intensity normalization for adc
            adc_normalized_file = os.path.join(output_path, 'adc_norm.nii.gz')
            utils.normalize_intensity(
                image_path=adc_registered_file,
                output_path=adc_normalized_file
            )
        # Prepare folders for nnU-Net
        nnunet_input_folder = os.path.join(path_nifti, subject_id, time_point, "nnUNet_input")
        os.makedirs(nnunet_input_folder, exist_ok=True)
        nnunet_output_folder = os.path.join(path_nifti, subject_id, time_point, "nnUNet_output")
        os.makedirs(nnunet_output_folder, exist_ok=True)

        # Map and copy files to nnUNet_input format
        file_mapping = {
            'flair_norm.nii.gz': f'{subject_id}_0000.nii.gz',
            't1_norm.nii.gz': f'{subject_id}_0001.nii.gz',
            't1ce_norm.nii.gz': f'{subject_id}_0002.nii.gz',
            't2_norm.nii.gz': f'{subject_id}_0003.nii.gz',
        }
        for original, renamed in file_mapping.items():
            original_path = os.path.join(output_path, original)
            if os.path.exists(original_path):
                renamed_path = os.path.join(nnunet_input_folder, renamed)
                shutil.copy(original_path, renamed_path)

        # Execute nnU-Net predict
        run_nnunet_predict(nnunet_input_folder, nnunet_output_folder)
        print(f"nnUNet prediction completed for: {nnunet_input_folder}")

        # Move segmentation results and perform cleanup
        for file in os.listdir(nnunet_output_folder):
            src_file = os.path.join(nnunet_output_folder, file)
            dst_file = os.path.join(output_path, "segmentation.nii.gz")
            shutil.move(src_file, dst_file)

        # Cleanup of temporary files
        shutil.rmtree(nnunet_input_folder)
        shutil.rmtree(nnunet_output_folder)
        renames = {
            'adc_norm.nii.gz': 'adc.nii.gz',
            'flair_norm.nii.gz': 'flair.nii.gz',
            't1_norm.nii.gz': 't1.nii.gz',
            't1ce_norm.nii.gz': 't1ce.nii.gz',
            't2_norm.nii.gz': 't2.nii.gz',
        }
        for original, new_name in renames.items():
            original_path = os.path.join(output_path, original)
            new_path = os.path.join(output_path, new_name)
            if os.path.exists(original_path):
                if os.path.exists(new_path):
                    os.remove(new_path)
                shutil.move(original_path, new_path)

        preserved_files = [
            os.path.join(output_path, 't1.nii.gz'),
            os.path.join(output_path, 't1ce.nii.gz'),
            os.path.join(output_path, 't2.nii.gz'),
            os.path.join(output_path, 'flair.nii.gz'),
            os.path.join(output_path, 'segmentation.nii.gz'),
        ]

        # adc.nii.gz is optional, verify its existence before adding it to the list
        adc_file_path = os.path.join(output_path, 'adc.nii.gz')
        if os.path.exists(adc_file_path):
            preserved_files.append(adc_file_path)

        # Delete all files that are not in the list of preserved ones
        all_files = glob.glob(os.path.join(output_path, '*'))
        for file in all_files:
            if file not in preserved_files:
                os.remove(file)

        print(f"Final clean-up and file renaming completed for subject {subject_id}, time point {time_point}.")

print("All processing completed.")