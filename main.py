import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


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
one_hot_train_labels1 = to_one_hot(train_labels)

# ベクトル化されたテストラベル
one_hot_test_labels1 = to_one_hot(test_labels)

# Kerasの機能を使ってone-hotエンコーディング（カテゴリエンコーディング）した場合
one_hot_train_labels2 = to_categorical(train_labels)
one_hot_test_labels2 = to_categorical(test_labels)

# 自作関数でone-hotエンコーディングした結果がKerasの関数を使った場合と同じ結果になる
# 今後はKerasの関数を使おう
print((one_hot_test_labels1 == one_hot_test_labels2).all())
print((one_hot_train_labels1 == one_hot_train_labels2).all())
