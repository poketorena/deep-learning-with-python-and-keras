import os, shutil

import numpy as np
from keras import layers
from keras import optimizers
from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import matplotlib.pyplot as plt

# 元のデータセットを展開したディレクトリへのパス
original_dataset_dir = "./dogs-vs-cats/train"

# より小さなデータセットを格納するディレクトリへのパス
base_dir = "./cats-and-dogs-small"
# os.mkdir(base_dir)

# 訓練データセット、検証データセット、テストデータセットを配置するディレクトリ
train_dir = os.path.join(base_dir, "train")
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, "validation")
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, "test")
# os.mkdir(test_dir)

# 訓練用の猫の画像を配置するディレクトリ
train_cats_dir = os.path.join(train_dir, "cats")
# os.mkdir(train_cats_dir)

# 訓練用の犬の画像を配置するディレクトリ
train_dogs_dir = os.path.join(train_dir, "dogs")
# os.mkdir(train_dogs_dir)

# 検証用の猫の画像を配置するディレクトリ
validation_cats_dir = os.path.join(validation_dir, "cats")
# os.mkdir(validation_cats_dir)

# 検証用の犬の画像を配置するディレクトリ
validation_dogs_dir = os.path.join(validation_dir, "dogs")
# os.mkdir(validation_dogs_dir)

# テスト用の猫の画像を配置するディレクトリ
test_cats_dir = os.path.join(test_dir, "cats")
# os.mkdir(test_cats_dir)

# テスト用の犬の画像を配置するディレクトリ
test_dogs_dir = os.path.join(test_dir, "dogs")
# os.mkdir(test_dogs_dir)

# 最初の1000個の猫画像をtrain_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    # shutil.copyfile(src, dst)

# 次の500個の猫画像をvalidation_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    # shutil.copyfile(src, dst)

# 次の500個の猫画像をtest_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    # shutil.copyfile(src, dst)

# 最初の1000個の犬画像をtrain_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    # shutil.copyfile(src, dst)

# 次の500個の犬画像をvalidation_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    # shutil.copyfile(src, dst)

# 次の500個の犬画像をtest_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    # shutil.copyfile(src, dst)

# コピーが成功したかチェックする（健全性チェック）
print("total training cat images:", len(os.listdir(train_cats_dir)))
print("total training dog images:", len(os.listdir(train_dogs_dir)))

print("total validation cat images:", len(os.listdir(validation_cats_dir)))
print("total validation dog images:", len(os.listdir(validation_dogs_dir)))

print("total test cat images:", len(os.listdir(test_cats_dir)))
print("total test dog images:", len(os.listdir(test_dogs_dir)))

conv_base = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(150, 150, 3))

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode="binary")
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # ジェネレータはデータを無限ループで生成するため
            # 画像を一通り処理したらbreakしなければならない
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)

test_features, test_labels = extract_features(test_dir, 1000)

# 平坦化する
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()

model.add(layers.Dense(256, activation="relu", input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

print(model.summary())

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=["acc"])

# モデルを適合
history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

# モデルを保存
model.save("cats_and_dogs_small_transfer_learning_fit_dense_only.h5")

# 訓練時の損失値を正解率をプロット
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

# 正解率をプロット
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

# 損失値をプロット
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
