# run_nnunet_predict.py
import os
import subprocess

def configure_environment():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Make paths using dir_path
    nnUNet_raw = os.path.join(dir_path, "my_nnunet", "nnUNet_raw")
    nnUNet_preprocessed = os.path.join(dir_path, "my_nnunet", "nnUNet_preprocessed")
    nnUNet_results = os.path.join(dir_path, "my_nnunet", "nnUNet_results")

    os.environ['nnUNet_raw'] = nnUNet_raw
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
    os.environ['nnUNet_results'] = nnUNet_results

def run(input_folder, output_folder):
    configure_environment()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    subprocess.run([
        'nnUNetv2_predict',
        '-d', 'Dataset001_BrainTumor',
        '-i', input_folder,
        '-o', output_folder,
        '-f', '0', '1', '2', '3', '4',
        '-tr', 'nnUNetTrainer',
        '-c', '3d_fullres',
        '-p', 'nnUNetPlans'
    ], env=env, capture_output=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run nnUNet Prediction')
    parser.add_argument('-i', '--input_folder', required=True, help='Input folder path')
    parser.add_argument('-o', '--output_folder', required=True, help='Output folder path')

    args = parser.parse_args()

    run(args.input_folder, args.output_folder)


