import os
import sys
import glob
import numpy as np
import cv2
import generate_feature


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# COSINE_THRESHOLD = 0.363
COSINE_THRESHOLD = 0.5
NORML2_THRESHOLD = 1.128

# 特徴を辞書と比較してマッチした最高値のユーザーとスコアを返す関数
def face_match(recognizer, feature1, dictionary):
    """
    input face_npy
    output face_name
    """
    max_score = float('-inf')
    max_user = None

    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)

        if score > max_score:
            max_score = score
            max_user = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    else:
        return True, (max_user, max_score)
    
def generate_aligned_faces(image):
    """
    input img
    output faces_img
    """
    # 入力サイズを指定する
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    # 顔を検出する
    result, faces = face_detector.detect(image)
    faces = faces if faces is not None else []
    # print("")

    return faces

def generate_feature_dictionary(image,face):
    """
    input face_img
    output face_npy
    """
    # 顔を切り抜き特徴を抽出する
    aligned_face = face_recognizer.alignCrop(image, face)
    feature = face_recognizer.feature(aligned_face)
    # print("")

    return feature

def sql_updater(data):
    import sqlite3
    import datetime
    import os
    """
    input face_names
    """
    date = datetime.datetime.now()
    d_today = str(datetime.date.today())
    min = int(date.hour)*60 + date.minute
    # data = [("user1",True),("user2",True),("user3",True)]


    dbname = os.path.join("..\\data",'history.db')
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()

    # テーブルが存在しなければ作成
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{d_today}'(min INTEGER PRIMARY KEY);")
    conn.commit()

    # カラムが存在しなければ作成
    cur.execute(f"PRAGMA table_info('{d_today}')")
    columns = [column[1] for column in cur.fetchall()]
    for user_name, bol in data:
        if user_name not in columns:
            cur.execute(f"ALTER TABLE '{d_today}' ADD COLUMN {user_name} TEXT DEFAULT False")
    conn.commit()

    # minが存在しなければ作成
    cur.execute(f"SELECT * FROM '{d_today}' WHERE min = {min}")
    if not cur.fetchone():
        cur.execute(f"REPLACE INTO '{d_today}' (min) values ({min})")
    conn.commit()

    # 対称の時間の行の値を更新する
    for user_name, bol in data:
        cur.execute(f"UPDATE '{d_today}' SET {user_name} = {bol} WHERE min = {min}")
    conn.commit()

    conn.close()


# 特徴を読み込む
dictionary = []
path_dir_path = os.path.join("..\\data","face_fig","**")
files = glob.glob(os.path.join(path_dir_path, "*.npy"), recursive=True)
for file in files:
    feature = np.load(file)
    user_id = os.path.basename(os.path.dirname(file))
    dictionary.append((user_id, feature))

# モデルを読み込む
model_dir_path = os.path.join("..\\data","model")
weights = os.path.join(model_dir_path, "yunet_n_640_640.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
weights = os.path.join(model_dir_path, "face_recognizer_fast.onnx")
face_recognizer = cv2.FaceRecognizerSF_create(weights, "")


def main(img_path):

    # face_figフォルダのnpyファイルを作成
    generate_feature.main()


    # キャプチャを開く
    capture = cv2.VideoCapture(img_path) # 画像ファイル
    # capture = cv2.VideoCapture("hoshino.mp4") # 動画
    # capture = cv2.VideoCapture(0) # カメラ
    if not capture.isOpened():
        print("no file or camera")
        exit()


    while True:
        # フレームをキャプチャして画像を読み込む
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            print("no image")
            break

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 顔を検出する
        faces = generate_aligned_faces(image)

        for face in faces:
            # 顔を切り抜き特徴を抽出する
            feature = generate_feature_dictionary(image,face)

            # 辞書とマッチングする
            result, user = face_match(face_recognizer, feature, dictionary)

            # 顔のバウンディングボックスを描画する
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # 認識の結果を描画する
            id, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

            # 認識の結果をDBに記録する
            if result:
                sql_updater([(user[0],True)])

        # 画像を表示する
        cv2.imshow("face recognition", image)
        cv2.imwrite(os.path.join(f"{img_path}_recognized.jpg"), image) # 動画・カメラの場合は要コメントアウト

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main("image.jpg")


# sqlite記述関数を作る。
# 識別結果は分単位で記録する
# 何分を取得する関数を探す
    
# 日にちでテーブルを用意する。ユーザーはカラムで管理する。1日を1440分で1440行とする。
# 全ての.npyファイルからベストスコアを結果とする方法で間違いが生じないか確認する。