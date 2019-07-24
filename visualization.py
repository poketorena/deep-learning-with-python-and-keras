from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

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

# Grad-CAMアルゴリズムの設定
# 予測ベクトルの「アフリカゾウ」エントリ
african_elephant_output = model.output[:, 386]

# VGG16の最後の畳み込み層であるblock5_conv3出力特徴マップ
last_conv_layer = model.get_layer("block5_conv3")

# block5_conv3の出力特徴マップでの「アフリカゾウ」クラスの勾配
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# 形状が(512,)のベクトル：
# 各エントリは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# 2頭のアフリカゾウのサンプル画像に基づいて、pooled_gradsと
# block5_conv3の出力特徴マップの値にアクセスするための関数
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# これら2つの値をNumPy配列として取得
pooled_grads_value, conv_layer_output_value = iterate([x])

# 「アフリカゾウ」クラスに関する「このチャネルの重要度」を
# 特徴マップ配列の各チャネルに掛ける
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 最終的な特徴マップのチャネルごとの平均値が
# クラスの活性化のヒートマップ
heatmap = np.mean(conv_layer_output_value, axis=-1)

# ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
