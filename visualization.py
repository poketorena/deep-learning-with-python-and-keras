from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# 出力層に全結合分類器が含まれていることに注意
# ここまでのケースでは、この分類器を削除している
model = VGG16(weights="imagenet")

# ターゲット画像へのローカルパス
img_path = "./creative_commons_elephant.jpg"

# ターゲット画像を読み込む：imgはサイズが224x224のPIL画像
img = image.load_img(img_path, target_size=(224, 224))

# VGG16モデルに合わせて入力画像を前処理

# xは形状が(224, 224, 3)のfloat32型のNumPy配列
x = image.img_to_array(img)

# この配列をサイズが(1, 224, 224, 3)のバッチに変換するために次元を追加
x = np.expand_dims(x, axis=0)

# バッチの前処理（チャネルごとに色を正規化）（RGBをBGRにしてR,G,Bの平均値を引いてる？）
x = preprocess_input(x)

preds = model.predict(x)

tmp1 = decode_predictions(preds, top=3)

tmp2 = tmp1[0]

print("Predicted:", decode_predictions(preds, top=3)[0])

print(np.argmax(preds[0]))
