from keras.models import Model
from keras import Input
from keras.datasets import mnist
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))

# モデルをKeras Functional APIで書いた場合
input_tensor = Input(shape=(28, 28, 1))

x = layers.Conv2D(filters=32,
                  kernel_size=(3, 3),
                  activation="relu")(input_tensor)

x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation="relu")(x)

x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation="relu")(x)

x = layers.Flatten()(x)

x = layers.Dense(units=64,
                 activation="relu")(x)

output_tensor = layers.Dense(units=10,
                             activation="softmax")(x)

model = Model(input_tensor, output_tensor)

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_loss:{test_loss}  test_acc:{test_acc}")
