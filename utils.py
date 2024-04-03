import os
import re
import nibabel as nib
import numpy as np
def normalize_intensity(image_path, output_path):
    # Load the image
    img = nib.load(image_path)
    data = img.get_fdata()

    # Compute the mean and standard deviation of the image's intensity
    mean_intensity = np.mean(data)
    std_intensity = np.std(data)

    # Perform the z-score normalization
    data_norm = (data - mean_intensity) / std_intensity

    # Save the normalized image
    img_norm = nib.Nifti1Image(data_norm, img.affine)
    nib.save(img_norm, output_path)


def remove_unwanted_suffixes(output_path):
    pattern = re.compile(r'^(t1|t1ce|t2|flair|adc|dwi)(_.+)?\.nii\.gz$')

    # Lista todos los archivos en el directorio de salida.
    for file in os.listdir(output_path):
        match = pattern.match(file)
        if match:
            correct_name = match.group(1) + '.nii.gz'
            original_path = os.path.join(output_path, file)
            new_path = os.path.join(output_path, correct_name)

            if original_path != new_path:
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(original_path, new_path)