import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),      # Changing Values here
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

a = model.get_weights()[0]          # Weights from input to first hidden layer
for i in range(0,10):               # Written for PROBLEM 2 to generate Image 2-D Matrix
  b = tf.reshape(a[:,i], [28,28])   # Choosing column i to procude 28x28 color matrix
  print 
  print "Image " + str(i+1) + ":"
  print tf.Session().run(b)         # Printing the same