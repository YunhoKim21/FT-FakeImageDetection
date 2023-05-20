import numpy as np


def DFT2D(image):
    m, n = image.shape

    a = np.arange(n).reshape(1, n).repeat(n, axis=0)
    a = a * a.T

    b = np.arange(m).reshape(1, m).repeat(m, axis=0)
    b = b * b.T

    a = np.exp(-2j * np.pi * a / n)
    b = np.exp(-2j * np.pi * b / m)

    return b @ image @ a.T

def IDFT2D(image):
    m, n = image.shape

    a = np.arange(n).reshape(1, n).repeat(n, axis=0)
    a = a * a.T

    b = np.arange(m).reshape(1, m).repeat(m, axis=0)
    b = b * b.T

    a = np.exp(2j * np.pi * a / n)
    b = np.exp(2j * np.pi * b / m)

    return np.real(b @ image @ a.T / (m * n))

if __name__ == '__main__':
    # for debug
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(DFT2D(array))
    print(IDFT2D(DFT2D(array)))

    print(np.fft.fft2(array))