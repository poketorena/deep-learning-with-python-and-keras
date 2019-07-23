# フィルタを可視化するための損失テンソルの定義
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights="imagenet", include_top=False)

layer_name = "block3_conv1"
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# 入力に関する損失関数の勾配を取得
# gradientsの呼び出しはテンソル（この場合はサイズ1）のリストを返す
# このため、最初の要素（テンソル）だけを保持する
grads = K.gradients(loss, model.input)[0]

# 勾配の正規化
# 除算の前に1e-5を足すことで、0による除算を回避
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 入力値をNumPy配列で受け取り、出力値をNumPy配列で返す関数
iterate = K.function([model.input], [loss, grads])

# さっそくテストしてみる
import numpy as np

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 確率的勾配降下法を使って損失値を最大化
# 最初はノイズが含まれたグレースケール画像を使用
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# 勾配上昇法を40ステップ実行
step = 1.  # 各勾配更新の大きさ
for i in range(40):
    # 損失値と勾配値を計算
    loss_value, grads_value = iterate([input_img_data])
    # 損失が最大になる方向に入力画像を調整
    input_img_data += grads_value * step


def deprocess_image(x):
    # テンソルを正規化：中心を0、標準偏差を0.1にする
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0,1]でクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


result = deprocess_image(input_img_data[0])

import matplotlib.pyplot as plt

plt.imshow(result)
plt.show()
