import numpy as np
import matplotlib.pyplot as plt
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


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# k分割交差検証
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

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

    # モデルを適合する
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=1)

    # maeを記録する
    mae_history = history.history["val_mean_absolute_error"]
    all_mae_histories.append(mae_history)

# k分割交差検証の平均スコアの履歴を構築
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

# 検証スコアのプロット
smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
