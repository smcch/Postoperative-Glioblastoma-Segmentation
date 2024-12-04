# Río Hortega Glioblastoma Segmentation - *RH-GlioSeg-nnUNet*
This repository contains the Python implementation of the paper: 
>Santiago Cepeda, Roberto Romero, Lidia Luque, Daniel García-Pérez, Guillermo Blasco, Luigi Tommaso Luppino, Samuel Kuttner, Olga Esteban-Sinovas, Ignacio Arrese, Ole Solheim, Live Eikenes, Anna Karlberg, Ángel Pérez-Núñez, Olivier Zanier, Carlo Serra, Victor E Staartjes, Andrea Bianconi, Luca Francesco Rossi, Diego Garbossa, Trinidad Escudero, Roberto Hornero, Rosario Sarabia, Deep Learning-Based Postoperative Glioblastoma Segmentation and Extent of Resection Evaluation: Development, External Validation, and Model Comparison, Neuro-Oncology Advances, 2024;, vdae199, https://doi.org/10.1093/noajnl/vdae199

**Ground truth segmentation**
![animated_ground_truth](https://github.com/smcch/Postoperative-Glioblastoma-Segmentation/assets/87584415/282185d7-9a47-4fd0-bd31-ab54c287b527)

**RH-GlioSeg-nnU-Net segmentation**
![animated_prediction](https://github.com/smcch/Postoperative-Glioblastoma-Segmentation/assets/87584415/0e7cb352-784c-4a5c-8b53-0e18886ce6f6)


This work presents a **fully automated pipeline** that incorporates the processing of multiparametric magnetic resonance imaging (MRI) and the automatic segmentation of **tumor subregions and surgical cavity in postoperative scans**. It includes the following stages:
- DICOM to NifTI conversion
- ADC computation
- Image registration
- Skull stripping
- Intensity normalization
- Tumor segmentation

## Installation
All required packages can be installed using the following command:
```bash
pip install -r requirements.txt
```
> [!NOTE]
> You should install PyTorch as described on their [website]([conda/pip](https://pytorch.org/get-started/locally/)) based on your system settings (OS, CUDA version, etc.)
> Due to Git file size limitations, the segmentation model must be downloaded from this (https://drive.google.com/file/d/17b6sKdyErUhhtcBRD2-oFOk-ORbGvDo1/view?usp=sharing) and extrated into the root path (the folder *my_nnunet* must be placed next to the file *main.py*).

## Data preparation
The pipeline accepts DICOM images (*.DCM) grouped as follows:
```
INPUT_FOLDER
├── Subject_1
│   ├── TimePoint_1
│   │   ├── dwi/adc (optional)
│   │   │   ├── *.DCM
│   │   │   ├── *.DCM
│   │   │   ├── ...
│   │   ├── flair
│   │   │   ├── ...
│   │   ├── t1
│   │   │   ├── ...
│   │   ├── t1ce
│   │   │   ├── ...
│   │   ├── t2
│   │   │   ├── ...
│   ├── TimePoint_2
│   ├── TimePoint_3
├── Subject_2
│   ├── ...
├── Subject_3
│   ├── ...
```
The input folder must be organized with a separate folder for each subject, named with an ID number. Within each subject folder, there should be a folder for each time point, also named with a number. As per convention, '0' denotes the preoperative scan, '1' represents the early postoperative scan, and subsequent numbers correspond to follow-up scans.

Within each time point folder, there should be four specific folders: *flair*, *t1*, *t1ce*, and *t2*, each containing a set of DICOM files. 

The user has the flexibility to either provide *dwi* DICOM files, allowing the pipeline to calculate ADC maps and integrate them into processing alongside other sequences, or directly supply DICOM files with ADC already calculated.

Also, you have to create an empty folder to output the results of the pipeline.

## Usage
### GUI
The pipeline comes with a grafical user interface (GUI) to easily run the pipeline.
### Command line
```bash
python.exe main.py -i INPUT_FOLDER -o OUTPUT_FOLDER
```
## Citations
If you find this pipeline useful for your academic purposes, please include the following citations:

- DICOM to NiFTI converter: `dcm2niix`, available at https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20220720
	- Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 2016;264:47-56. doi:10.1016/j.jneumeth.2016.03.001.
- REGISTRATION: `SimpleElastix`: A user-friendly, multi-lingual library for medical image registration, available at https://simpleelastix.github.io/
	- Marstal K, Berendsen F, Staring M, Klein S. SimpleElastix: A user-friendly, multi-lingual library for medical image registration. Computer Methods and Programs in Biomedicine. 2018;154:103-119. doi:10.1016/j.cmpb.2017.11.010.
 - SKULL STRIPPING: `SynthStrip`: A tool for brain MRI skull stripping and synthetic MRI generation, available at [link_to_synthstrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
	- Engwer C, Schwaiger BJ, Maaß A, Würfl T, Langkammer C, Haynor DR, Schölkopf B, Golland P, Menze BH, Ronneberger O. SynthStrip: A tool for brain MRI skull stripping and synthetic MRI generation. Medical Image Analysis. 2021;71:102091. doi:10.1016/j.media.2021.102091.
- SEGMENTATION: `nnUNet`: A framework for automated segmentation of medical image data, available at [link_to_nnUNet](https://github.com/MIC-DKFZ/nnUNet)
	- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: Breaking the Spell on Successful Medical Image Segmentation. arXiv:1809.10486. 2018.


## License
Creative Commons Attribution-NonCommercial License: This repository is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. This license allows others to freely use, modify, and distribute the software for non-commercial purposes only. You are granted the right to use this software for personal, educational, and non-profit projects, but commercial use is not permitted without explicit permission. For more details, please refer to the LICENSE file.
