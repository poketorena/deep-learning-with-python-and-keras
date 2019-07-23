# フィルタを可視化するための損失テンソルの定義
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights="imagenet", include_top=False)
model.summary()


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


def generate_pattern(layer_name, filter_index, size=150):
    # ターゲット層のn番目のフィルタの活性化を最大化する損失関数を構築
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # この損失関数を使って入力画像の勾配を計算
    grads = K.gradients(loss, model.input)[0]

    # 正規化トリック：勾配を正規化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて損失値と勾配値を返す関数
    iterate = K.function([model.input], [loss, grads])

    # 最初はノイズが含まれたグレースケール画像を使用
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 勾配上昇法を40ステップ実行
    step = 1.

    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


# mainプログラム
layers = ["block1_conv1",
          "block1_conv2",

          "block2_conv1",
          "block2_conv2",

          "block3_conv1",
          "block3_conv2",
          "block3_conv3",

          "block4_conv1",
          "block4_conv2",
          "block4_conv3",

          "block5_conv1",
          "block5_conv2",
          "block5_conv3", ]
for layer_name in layers:
    size = 64
    margin = 5

    # 結果を格納する空（黒）の画像
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype=np.uint8)

    for i in range(8):  # resultsグリッドの行を順番に処理
        for j in range(8):  # resultsグリッドの列を順番に処理
            # layer_nameのフィルタi+(j*8)のパターンを生成
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # resultsグリッドの短形（i, j）に結果を配置
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img

    # resultsグリッドを表示
    plt.figure(figsize=(20, 20))
    plt.title(layer_name)
    plt.imshow(results)
    plt.savefig(layer_name)
    plt.show()
