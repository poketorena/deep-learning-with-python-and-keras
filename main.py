import numpy as np
from keras import models
from keras.datasets import boston_housing
from keras import layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 訓練データの形状
print(train_data.shape)
# テストデータの形状
print(test_data.shape)
# 住宅価格（単位は1000ドル）
# 住宅価格は10000ドルから50000ドルの間（1970年台は不動産バブルだから住宅が安かった！）
print(train_targets)

# データの正規化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    # 同じモデルを複数回インスタンス化するため
    # モデルをインスタンス化するための関数を使用
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu",
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# k分割交差検証
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("processing fold #", i)

    # 検証データの準備：フォールドiのデータ
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    # 訓練データの準備：残りのフォールドのデータ
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    # Kerasモデルを構築（コンパイル済み）
    model = build_model()

    # モデルをサイレントモード（verbose=0で適合する。これにより標準出力に出てくるテキストが少なくなる）
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=1)

    # モデルを検証データで評価
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)

# 結果の表示
print(all_scores)
print(np.mean(all_scores))
