from keras.datasets import imdb
import numpy as np


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
