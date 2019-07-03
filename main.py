import copy
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))
print(train_data[10])

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# インデックスのオフセットとして3が指定されているのは
# 0、1、2がそれぞれ「パディング」、「シーケンスの開始」、「不明」の
# インデックスとして予約されているためであることに注意
decoded_newswire = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)

# デコードしたニュースを表示
print(decoded_newswire)

# サンプルに関連付けられているラベル
print(train_labels[10])

# 訓練データのベクトル化
x_train = vectorize_sequences(train_data)

# テストデータのベクトル化
x_test = vectorize_sequences(test_data)

# ベクトル化された訓練ラベル
# one_hot_train_labels1 = to_one_hot(train_labels)

# ベクトル化されたテストラベル
# one_hot_test_labels1 = to_one_hot(test_labels)

# ラベルをベクトル化しないで整数のテンソルとしてキャストする場合
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)

# Kerasの機能を使ってone-hotエンコーディング（カテゴリエンコーディング）した場合
one_hot_train_labels1 = to_categorical(train_labels)
one_hot_test_labels1 = to_categorical(test_labels)

# 自作関数でone-hotエンコーディングした結果がKerasの関数を使った場合と同じ結果になる
# 今後はKerasの関数を使おう
# print((one_hot_test_labels1 == one_hot_test_labels2).all())
# print((one_hot_train_labels1 == one_hot_train_labels2).all())

# モデルの作成
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 検証データセットの設定
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels1[:1000]
partial_y_train = one_hot_train_labels1[1000:]

# モデルの訓練
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=8,
                    batch_size=128,
                    validation_data=(x_val, y_val))

# 結果の表示
results = model.evaluate(x_test, one_hot_test_labels1)
print("学習結果", results)

# ランダムをベースラインとした結果と比較する
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
print("ランダムの場合", float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels))

# 新しいデータで予測値を生成する
predictions = model.predict(x_test)

# 各エントリの長さ（46クラスの分類問題だから46になる）
print(predictions[0].shape)
# このベクトルの係数を合計すると1になる（確率分布だから）
print(np.sum(predictions[0]))
# 最も大きなエントリが予測されたクラス（この場合は3）
print(np.argmax(predictions[0]))

# 訓練データと検証データでの損失値をプロット
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 訓練データと検証データでの正解率のプロット
plt.clf()  # 図を消去

acc = history.history["acc"]
val_acc = history.history["val_acc"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
