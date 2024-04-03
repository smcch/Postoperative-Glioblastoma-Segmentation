import SimpleITK as sitk
from types import SimpleNamespace
import os

class run():
    def __init__(self, **kwargs):
        defaultKwargs = {}
        args = SimpleNamespace(**{**defaultKwargs, **kwargs})

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(sitk.ReadImage(args.reference))
        elastixImageFilter.SetMovingImage(sitk.ReadImage(args.in_file))

        parameterMapVector = sitk.VectorOfParameterMap()
        # Apply translation
        parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
        # Apply rigid
        parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
        # Apply affine
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        # Remove temporary files
        for i in [0, 1, 2]:
            os.remove(f'TransformParameters.{i}.txt')
        # Save registered image
        sitk.WriteImage(elastixImageFilter.GetResultImage(), args.out_file)