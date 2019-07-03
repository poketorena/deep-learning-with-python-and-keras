from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))
print(train_data[10])

