from types import SimpleNamespace
import re
import io
import nibabel as nib
import numpy as np

from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from medpy.filter import otsu
from medpy.core.exceptions import ArgumentError
from medpy.filter.binary import largest_connected_component

# FIX: ensure compatibility on different operative systems
np.float = np.float64
np.bool = np.bool_
##########################################################

class run():
    def __init__(self, **kwargs):
        defaultKwargs = {}
        args = SimpleNamespace(**{**defaultKwargs, **kwargs})

        # Read and process bvals
        with open(args.bvalPath, 'r') as f:
            content = f.read()
        munged_content = io.StringIO(re.sub(r'(\t|,)', ' ', content))
        vals = np.squeeze(np.loadtxt(munged_content))

        # Assume b0 as 0 and find the highest b value if there are more than 2 values
        b0 = 0
        if len(vals) > 2:
            bX = max([val for val in vals if val != b0])
        elif len(vals) == 2:
            bX = max(vals)
        else:
            raise ValueError("At least 2 'b' values were expected, including 'b0'.")

        # Identify the indices for b0 and bX
        idxB0 = np.where(vals == b0)[0][0]
        idxBX = np.where(vals == bX)[0][0]

        # Load DWI data
        dwi = nib.load(args.dwiPath)
        dwi_data = dwi.get_fdata()
        b0img = dwi_data[:, :, :, idxB0]
        bximg = dwi_data[:, :, :, idxBX]

        # Convert to float for processing
        b0img = b0img.astype(np.float64)
        bximg = bximg.astype(np.float64)

        # Threshold using Otsu
        b0thr = otsu(b0img, 32) / 20.
        bxthr = otsu(bximg, 32) / 20.
        if b0thr <= 0:
            raise ArgumentError("Image 'b0' contains negative values")
        if bxthr <= 0:
            raise ArgumentError("Image 'bx' contains negative values")

        # Create the mask
        mask = binary_fill_holes(b0img > b0thr) & binary_fill_holes(bximg > bxthr)
        mask = binary_erosion(mask, iterations=1)
        mask = largest_connected_component(mask)
        mask = binary_dilation(mask, iterations=1)

        # Compute ADC
        adc = np.zeros(b0img.shape, dtype=b0img.dtype)
        mask &= (b0img != 0)
        adc[mask] = -1. * bX * np.log(bximg[mask] / b0img[mask])
        adc[adc < 0] = 0

        # Save ADC image
        adc_img = nib.Nifti1Image(adc, dwi.affine)
        nib.save(adc_img, args.adcPath)
