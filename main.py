import dlib
import cv2
import os
import numpy as np

from ULFD import ULFD_face_detector

ORIGINAL_WIDTH = 2316
ORIGINAL_HEIGHT = 3088

PRINTS = False


def align_face(face_img):

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

    face_detector = ULFD_face_detector()
    predictor_path = "saved_models/shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    cords, scores = face_detector(face_img)
    if cords is None:
        return None
    landmarks = predictor(gray, dlib.rectangle(cords[0], cords[1], cords[2], cords[3]))

    left_eye = np.array([(landmarks.part(2).x, landmarks.part(2).y),
                         (landmarks.part(3).x, landmarks.part(3).y)], dtype="double")
    right_eye = np.array([(landmarks.part(0).x, landmarks.part(0).y),
                          (landmarks.part(1).x, landmarks.part(1).y)], dtype="double")

    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")

    if PRINTS:
        print(f"left eye center: {left_eye_center}")
        print(f"right eye center: {right_eye_center}")

    d_y = right_eye_center[1] - left_eye_center[1]
    d_x = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(d_y, d_x))

    if PRINTS:
        print(f"angle: {angle}")

    desired_left_eye_x = 0.33 * ORIGINAL_WIDTH
    desired_right_eye_x = 0.67 * ORIGINAL_WIDTH

    if PRINTS:
        print(f"desired_left_eye_x: {desired_left_eye_x}")
        print(f"desired_right_eye_x: {desired_right_eye_x}")

    dist = np.sqrt((d_x ** 2) + (d_y ** 2))
    desired_dist = desired_right_eye_x - desired_left_eye_x
    scale = desired_dist / dist

    if PRINTS:
        print(f"dist: {dist}")
        print(f"desired_dist: {desired_dist}")
        print(f"scale: {scale}")

    eyes_center = (float((left_eye_center[0] + right_eye_center[0]) // 2),
                   float((left_eye_center[1] + right_eye_center[1]) // 2))

    if PRINTS:
        print(f"eyes_center: {eyes_center}")

    rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    t_x = ORIGINAL_WIDTH * 0.5
    t_y = ORIGINAL_HEIGHT * 0.4

    if PRINTS:
        print(f"t_x: {t_x}")
        print(f"t_y: {t_y}")
    rot_matrix[0, 2] += (t_x - eyes_center[0])
    rot_matrix[1, 2] += (t_y - eyes_center[1])

    aligned_face = cv2.warpAffine(face_img, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_face


# main
input_dir = "images/quent1500/"
output_dir = "images/quent2500/"

for img_file in os.listdir(input_dir):
    print(f"processing {img_file}")
    img_path = os.path.join(input_dir, img_file)
    image = cv2.imread(img_path)

    aligned = align_face(image)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, aligned)
