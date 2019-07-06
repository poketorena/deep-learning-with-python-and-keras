import os, shutil
from keras import layers
from keras import optimizers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 元のデータセットを展開したディレクトリへのパス
original_dataset_dir = "./dogs-vs-cats/train"

# より小さなデータセットを格納するディレクトリへのパス
base_dir = "./cats-and-dogs-small"
os.mkdir(base_dir)

# 訓練データセット、検証データセット、テストデータセットを配置するディレクトリ
train_dir = os.path.join(base_dir, "train")
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, "validation")
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, "test")
os.mkdir(test_dir)

# 訓練用の猫の画像を配置するディレクトリ
train_cats_dir = os.path.join(train_dir, "cats")
os.mkdir(train_cats_dir)

# 訓練用の犬の画像を配置するディレクトリ
train_dogs_dir = os.path.join(train_dir, "dogs")
os.mkdir(train_dogs_dir)

# 検証用の猫の画像を配置するディレクトリ
validation_cats_dir = os.path.join(validation_dir, "cats")
os.mkdir(validation_cats_dir)

# 検証用の犬の画像を配置するディレクトリ
validation_dogs_dir = os.path.join(validation_dir, "dogs")
os.mkdir(validation_dogs_dir)

# テスト用の猫の画像を配置するディレクトリ
test_cats_dir = os.path.join(test_dir, "cats")
os.mkdir(test_cats_dir)

# テスト用の犬の画像を配置するディレクトリ
test_dogs_dir = os.path.join(test_dir, "dogs")
os.mkdir(test_dogs_dir)

# 最初の1000個の猫画像をtrain_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 次の500個の猫画像をvalidation_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 次の500個の猫画像をtest_cats_dirにコピー
fnames = [f"cat.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 最初の1000個の犬画像をtrain_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 次の500個の犬画像をvalidation_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 次の500個の犬画像をtest_dogs_dirにコピー
fnames = [f"dog.{i}.jpg" for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# コピーが成功したかチェックする（健全性チェック）
print("total training cat images:", len(os.listdir(train_cats_dir)))
print("total training dog images:", len(os.listdir(train_dogs_dir)))

print("total validation cat images:", len(os.listdir(validation_cats_dir)))
print("total validation dog images:", len(os.listdir(validation_dogs_dir)))

print("total test cat images:", len(os.listdir(test_cats_dir)))
print("total test dog images:", len(os.listdir(test_dogs_dir)))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

print(model.summary())

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])

# 全ての画像を1/255でスケーリング
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # ターゲットディレクトリ
    target_size=(150, 150),  # すべての画像サイズを150x150に変更
    batch_size=20,  # バッチサイズ
    class_mode="binary"  # binary_crossentropyを使用するため、二値のラベルが必要
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # ターゲットディレクトリ
    target_size=(150, 150),  # すべての画像サイズを150x150に変更
    batch_size=20,  # バッチサイズ
    class_mode="binary"  # binary_crossentropyを使用するため、二値のラベルが必要
)

# ImageDataGeneratorのテスト
for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    plt.imshow(data_batch[0])
    plt.show()
    break

# バッチジェネレータを使ってモデルを適合
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

# モデルを保存
model.save("cats_and_dogs_small_1.h5")

# 訓練時の損失値を正解率をプロット
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# 正解率をプロット
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.show()

# 損失値をプロット
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
