import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import azimuthalAverage, high_pass_filter
from fft import DFT2D, IDFT2D

def pipeline(img_path, root = 'images'):
    img = cv2.imread(img_path, 0) / 256

    img = cv2.resize(img, (224, 224))

    img_name = img_path.split('/')[-1].split('.')[0]

    cv2.imwrite(os.path.join(root, 'resized', img_name + '.jpg'), img * 256)

    dft2d = DFT2D(img)

    dft2d = np.fft.fftshift(dft2d)

    magnitude_spectrum = 20 * np.log1p(np.abs(dft2d))
    cv2.imwrite(os.path.join(root, 'spectrum', img_name + '.jpg'), magnitude_spectrum)

    radial_profile = azimuthalAverage(magnitude_spectrum, normalized=True)

    plt.figure()
    plt.plot(radial_profile)
    plt.title('Radial Profile')
    plt.savefig(os.path.join(root, 'azmutal', img_name + '.jpg'))
    plt.close()


    dft2d = high_pass_filter(dft2d, radius=45)

    magnitude_spectrum = 20 * np.log1p(np.abs(dft2d))
    cv2.imwrite(os.path.join(root, 'high_passed_spectrum', img_name + '.jpg'), magnitude_spectrum)

    dft2d = np.fft.ifftshift(dft2d)

    idft2d = IDFT2D(dft2d) * 256
    
    cv2.imwrite(os.path.join(root, 'high_passed', img_name + '.jpg'), idft2d)

    return radial_profile

if __name__ == '__main__':
    pipeline(f'images/resized/000.jpg')