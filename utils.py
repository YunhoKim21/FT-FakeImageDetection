import numpy as np 
from tqdm import tqdm 

def azimuthalAverage(image, normalized=False):
    h, w = image.shape 
    center = h // 2, w // 2
    y, x = np.indices((h, w))

    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    distances = np.round(distances).astype(np.int)

    max_distance = int(np.max(distances))

    data = [[] for i in range(max_distance + 1)]

    for i in range(h):
        for j in range(w):
            data[distances[i, j]].append(image[i, j])
    
    data = np.array([np.mean(data[i]) for i in range(max_distance + 1)])
    return data / np.max(data) if normalized else data 



def high_pass_filter(dft2d, radius=10):
    
    M, N = dft2d.shape
    for i in range(M):
        for j in range(N):
            if np.sqrt((i - M/2)**2 + (j - N/2)**2) < radius:
                dft2d[i, j] = 0

    return dft2d

