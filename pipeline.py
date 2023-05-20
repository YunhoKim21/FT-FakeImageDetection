import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import azimuthalAverage, high_pass_filter
from fft import DFT2D, IDFT2D

def pipeline(img_path, root = 'images'):
    # load the image as grayscale
    img = cv2.imread(img_path, 0) / 256

    img = cv2.resize(img, (224, 224))

    img_name = img_path.split('/')[-1].split('.')[0]

    # save the image
    cv2.imwrite(os.path.join(root, 'resized', img_name + '.jpg'), img * 256)

    # apply 2D DFT
    dft2d = DFT2D(img)

    # apply fftshift to bring the zero-frequency component to the center of the spectrum
    dft2d = np.fft.fftshift(dft2d)

    # save dft2d as image
    magnitude_spectrum = 20 * np.log1p(np.abs(dft2d))
    cv2.imwrite(os.path.join(root, 'spectrum', img_name + '.jpg'), magnitude_spectrum)

    # apply azimuthalAverage to get the radial profile
    radial_profile = azimuthalAverage(magnitude_spectrum, normalized=True)

    # save radial_profile as image
    plt.figure()
    plt.plot(radial_profile)
    plt.title('Radial Profile')
    plt.savefig(os.path.join(root, 'azmutal', img_name + '.jpg'))
    plt.close()


    # apply high pass filter
    dft2d = high_pass_filter(dft2d, radius=45)

    # plot spectrum after high pass filter
    magnitude_spectrum = 20 * np.log1p(np.abs(dft2d))
    cv2.imwrite(os.path.join(root, 'high_passed_spectrum', img_name + '.jpg'), magnitude_spectrum)

    # apply inverse fftshift
    dft2d = np.fft.ifftshift(dft2d)

    # apply 2D IDFT
    idft2d = IDFT2D(dft2d) * 256
    #print(np.max(idft2d), np.min(idft2d))

    # save idft2d as image
    
    cv2.imwrite(os.path.join(root, 'high_passed', img_name + '.jpg'), idft2d)

    return radial_profile

if __name__ == '__main__':
    pipeline(f'images/resized/000.jpg')