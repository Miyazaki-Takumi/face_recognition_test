import os
import numpy as np
import cv2
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def features_detection(img_path):
    # 画像を開く
    image = cv2.imread(img_path)
    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    model_dir = "model"
    weights = os.path.join(model_dir, "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 特徴を抽出する
    face_feature = face_recognizer.feature(image)

    return face_feature


def main():

    dir_path = os.path.join(".\\data","face_fig")

    # ディレクトリに含まれるすべての.jpgファイル
    files = glob.glob(os.path.join(dir_path, "**", "*.jpg"), recursive=True)

    for file in files:
        print(file)
        # .jpgから特徴点をnpで計算
        face_feature = features_detection(file)

        dictionary = os.path.splitext(os.path.dirname(file))[0]
        basename = os.path.splitext(os.path.basename(file))[0]
        save_path = os.path.join(dictionary, f"{basename}.npy")
        print(save_path)

        # 同ファイル名で.npyファイルを保存
        np.save(save_path, face_feature)

        


if __name__ == '__main__':
    main()