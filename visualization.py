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

# 個のも出つの訓練に使用された入力が次の方法で前処理されていることに注意
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
