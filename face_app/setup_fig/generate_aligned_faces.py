import os
import uuid
import argparse
import numpy as np
import cv2


os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main(img_path):
    # 引数をパースする


    # 画像を開く
    image = cv2.imread(img_path)
    if image is None:
        print("cant open")
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    model_dir_path = "model"
    weights = os.path.join(model_dir_path, "yunet_n_640_640.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    weights = os.path.join(model_dir_path, "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 入力サイズを指定する
    height, width, _ = image.shape
    print("height:", height, "width:", width)
    face_detector.setInputSize((width, height))

    # 顔を検出する
    _, faces = face_detector.detect(image)

    # 検出された顔を切り抜く
    aligned_faces = []
    if faces is not None:
        print(f"aligned {len(faces)} faces")
        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            aligned_faces.append(aligned_face)
    else:
        print(f"no face")
    # 画像を表示、保存する
    save_dir_path = "output_face_img"
    for i, aligned_face in enumerate(aligned_faces):
        # cv2.imshow("aligned_face {:03}".format(i + 1), aligned_face) #window表示
        cv2.imwrite(os.path.join(save_dir_path, f"face_{uuid.uuid1()}.jpg"), aligned_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")

if __name__ == '__main__':
    for file in os.listdir("input_img"):
        img_path = os.path.join("input_img",file)
        main(img_path)