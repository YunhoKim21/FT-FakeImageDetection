# Utilizing Fourier Transformations for the Detection of Fake Images

Implementation of Fourier transform and other methods for detecting fake images. 

## Usage 

```
python run.py --root {root_folder}
```

Folder should be like this:

```
root_folder
├── src
    ├── 000.jpg
    ├── 001.jpg
    ├── 002.jpg
    ├── 003.jpg
    ├── 004.jpg
    ├── 005.jpg
    ├── 006.jpg
```

By running run.py, you get Fourier transform, High passed image, power spectrum, and classification by eye activation, which is printed out like below. 

```
eye activation of 000.jpg is 4.878, classified : Real
eye activation of 001.jpg is 0.634, classified : Fake
eye activation of 002.jpg is 1.254, classified : Fake
eye activation of 003.jpg is 5.046, classified : Real
eye activation of 004.jpg is 3.811, classified : Real
eye activation of 005.jpg is 2.530, classified : Fake
eye activation of 006.jpg is 4.105, classified : Real
```