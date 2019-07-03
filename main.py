from keras.datasets import reuters

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
