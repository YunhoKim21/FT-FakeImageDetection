from pipeline import pipeline
import os 
from tqdm import tqdm 
from argparse import ArgumentParser
import cv2

'''
Usage : python run.py --root {root_folder}

Folder should be like this:

root_folder
├── src
    ├── 000.jpg
    ├── 001.jpg
    ├── 002.jpg
    ├── 003.jpg
    ├── 004.jpg
    ├── 005.jpg
    ├── 006.jpg

run.py
'''

parser = ArgumentParser()
parser.add_argument('--root', type=str, default='PA_data', help='root directory')
args = parser.parse_args()

root = args.root 

if not os.path.exists(os.path.join(root, 'resized')):
    os.makedirs(os.path.join(root, 'resized'))

if not os.path.exists(os.path.join(root, 'spectrum')):
    os.makedirs(os.path.join(root, 'spectrum'))

if not os.path.exists(os.path.join(root, 'azmutal')):
    os.makedirs(os.path.join(root, 'azmutal'))

if not os.path.exists(os.path.join(root, 'high_passed_spectrum')):
    os.makedirs(os.path.join(root, 'high_passed_spectrum'))

if not os.path.exists(os.path.join(root, 'high_passed')):
    os.makedirs(os.path.join(root, 'high_passed'))

if not os.path.exists(os.path.join(root, 'eye_detected')):
    os.makedirs(os.path.join(root, 'eye_detected'))


image_files = sorted(os.listdir(os.path.join(root, 'src')))

for image_file in tqdm(image_files, desc='Performing FT, IFT, and Azimuthal Average'):
    radial_profile = pipeline(os.path.join(root, 'src', image_file), root)

from eye_activation import eye_activation
from constants import eye_thresh

print('* Eye Activation *\n')
print(f'classifies image based on eye activation threshold value, {eye_thresh}\n')
for image_file in image_files: 
    activation, image = eye_activation(os.path.join(root, 'resized', image_file),\
                                        os.path.join(root, 'high_passed', image_file),\
                                          debug=True )
    print(f'eye activation of {image_file} is {activation:.3f}, classified : {"Real" if activation > eye_thresh else "Fake"}')
    cv2.imwrite(os.path.join(root, 'activation', image_file), image)