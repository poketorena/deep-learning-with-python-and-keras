from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 訓練データの形状
print(train_data.shape)
# テストデータの形状
print(test_data.shape)
# 住宅価格（単位は1000ドル）
# 住宅価格は10000ドルから50000ドルの間（1970年台は不動産バブルだから住宅が安かった！）
print(train_targets)
