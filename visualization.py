import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import models
import matplotlib.pyplot as plt

# 保存したモデルを読み込む
model = load_model("cats_and_dogs_small_2.h5")

# 読み込んだモデルを出力する
print(model.summary())

# 単一の画像を前処理

# 5.2.2項でsmallデータセットを格納したディレクトリへのパスであることに注意
img_path = "cats-and-dogs-small/test/cats/cat.1700.jpg"

# この画像を4次元テンソルとして前処理
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# このモデルの訓練に使用された入力が次の方法で前処理されていることに注意
img_tensor /= 255.

# 形状は(1, 150, 150, 3)
print(img_tensor.shape)

# テスト画像を表示
plt.imshow(img_tensor[0])
plt.show()

# 入力テンソルと出力テンソルのリストに基づいてモデルをインスタンス化する
# 出力側の8つの層から出力を抽出
layer_outputs = [layer.output for layer in model.layers[:8]]

# 特定の層の入力をもとに、これらの出力を返すモデルを作成
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# モデルを予測モードで実行
# 5つのNumpy配列（層の活性化ごとに1つ）のリストを返す
activations = activation_model.predict(img_tensor)

# 猫の入力画像に対する最初の畳み込み層の活性化
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 元のモデルの最初の層の活性化の3番目のチャネルをプロットする
plt.matshow(first_layer_activation[0, :, :, 3], cmap="viridis")
plt.show()

plt.matshow(first_layer_activation[0, :, :, 30], cmap="viridis")
plt.show()

# 中間層の活性化ごとにすべてのチャネルを可視化
# プロットの一部として使用する層の名前
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# 特徴マップを表示
for layer_name, layer_activation in zip(layer_names, activations):
    # 特徴マップに含まれている特徴量の数
    n_features = layer_activation.shape[-1]

    # 特徴マップの形状(1, size, size, n_features)
    size = layer_activation.shape[1]

    # この行列で活性化のチャネルをタイル表示
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 各フィルタを1つの大きな水平グリッドタイルで表示
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            # 特徴量の見た目をよくするための後処理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image

    # グリッドを表示
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect="auto", cmap="viridis")

plt.show()

predict_result = model.predict(img_tensor)
print()
