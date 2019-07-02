from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_images.shape", train_images.shape)
print("len(train_labels)", len(train_labels))
print("train_labels", train_labels)

print("test_images", test_images.shape)
print("len(test_labels)", len(test_labels))
print("test_labels", test_labels)

from keras import models
from keras import layers

# Denseは全結合層（普通のニューラルネットワーク）のこと
# network = models.Sequential()
# network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation="softmax"))
# model = network

# Functional APIを使って書く場合
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(512, activation="relu")(input_tensor)
output_tensor = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)

# モデルをコンパイルする
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 0-1の値に変換する
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_iamges = test_images.astype("float32") / 255

# ラベルをカテゴリ値でエンコードする
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 訓練する
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# テストデータで検証する
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_loss:", test_loss)
print("test_acc:", test_acc)
