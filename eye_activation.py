from mtcnn import MTCNN
import cv2
import numpy as np

# Load the detector
detector = MTCNN()

def eye_activation(image_path, activation_path, debug=False):
    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    activation = cv2.imread(activation_path)

    # Detect faces in the image
    result = detector.detect_faces(image)

    eye_regions = []
    for person in result:
        bounding_box = person['box']
        keypoints = person['keypoints']

        eye_width = int(bounding_box[2] * 0.4)
        eye_height = int(bounding_box[3] * 0.2)

        left_eye_region = [int(keypoints['left_eye'][0]-eye_width//2), int(keypoints['left_eye'][1]-eye_height//2),
                        int(keypoints['left_eye'][0]+eye_width//2), int(keypoints['left_eye'][1]+eye_height//2)]
        right_eye_region = [int(keypoints['right_eye'][0]-eye_width//2), int(keypoints['right_eye'][1]-eye_height//2),
                            int(keypoints['right_eye'][0]+eye_width//2), int(keypoints['right_eye'][1]+eye_height//2)]

        eye_regions.append(left_eye_region)
        eye_regions.append(right_eye_region)

    if debug:
        output_image = image 
    

    area, s = 0, 0
    for eye_region in eye_regions:
        area += (eye_region[2] - eye_region[0]) * (eye_region[3] - eye_region[1])
        s += np.sum(activation[eye_region[1]:eye_region[3], eye_region[0]:eye_region[2]])
        if debug:
            cv2.rectangle(output_image, (eye_region[0], eye_region[1]), (eye_region[2], eye_region[3]), (0, 255, 0), 2)


    return s / area, output_image if debug else s / area 