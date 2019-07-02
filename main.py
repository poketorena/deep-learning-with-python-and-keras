from keras.datasets import imdb
from keras import models, optimizers, losses, metrics
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimention=10000):
    # 形状が(len(sequences), dimention)の行列を作成し、0で埋める
    results = np.zeros((len(sequences), dimention))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # results[i]のインデックスを1に設定

    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 訓練データのベクトル化
x_train = vectorize_sequences(train_data)
# テストデータのベクトル化
x_test = vectorize_sequences(test_data)

print(x_train[0])
print(x_test[0])

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# 最も単純なモデルのコンパイル
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])

# オプティマイザの設定
#
# オプティマイザのパラメータを指定したい場合はoptimizerパラメータに
# 引数としてオプティマイザクラスのインスタンスを設定する
#
# 独自の損失関数や指標関数を使用したい場合はlossパラメータか
# metricsパラメータに引数として関数オブジェクトを指定する
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

# 検証データセットの設定
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# モデルの訓練
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 訓練データと検証データの損失値をプロット
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)

# "bo"は"blue dot"（青のドット）を意味する
plt.plot(epochs, loss_values, "bo", label="Training loss")
# "b"は"solid blue line"（青の実線）を意味する
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epocs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 訓練データと検証データでの正解率をプロット

# 図を消去
plt.clf()

acc = history_dict["acc"]
val_acc = history_dict["val_acc"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
