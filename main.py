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
