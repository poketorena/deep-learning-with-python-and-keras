from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])

print(max([max(sequence) for sequence in train_data]))

# word_indexは単語を整数のインデックスにマッピングする辞書
word_index = imdb.get_word_index()

# 整数のインデックスを単語にマッピング
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# レビューを出コード：インデックスのオフセットとして3が指定されているのは、
# 0、1、2がそれぞれ「パディング」、「シーケンスの開始」、「不明」の
# インデックスとして予約されていることであることに注意
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)

# デコードしたレビューの内容を表示
print(decoded_review)
