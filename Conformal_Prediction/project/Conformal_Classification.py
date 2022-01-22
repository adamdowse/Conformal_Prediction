import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

#Classification version of conformal inference(produce sets of labels)

# Construct a tf.data.Dataset
train_ds,test_ds,val_ds = tfds.load('mnist', split=['train[:60%]','train[60%:80%]','train[80%:]'], shuffle_files=True,as_supervised=True)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

train_ds = train_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

#build the model
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = build_model()
history = model.fit(
    train_ds.batch(128),
    epochs=8,
    validation_data=test_ds.batch(128),
)

def plot_loss(history,name):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(name)
    plt.close()

plot_loss(history,'Classification_Simple_Loss')


#conformal prediction 
alpha=0.1
# score of the validation set is the softmax score of the correct class
#better way of doing this is to do it in batches for speed reasons
scores = []
batch_size = 128
for (image,label) in val_ds.batch(batch_size):
    output = model.predict(image)
    for i in range(tf.size(label)):
        l = label[i]
        scores.append(1-output[i,l])
n = len(list(val_ds))
qhat = 1-np.percentile(scores,(((n+1)*(1-alpha))/n)*100)
print("Basic qhat = ",qhat)

plt.hist(scores)
plt.vlines(qhat,0,200, color='r')
plt.savefig("hist_class_simple")
plt.close()

#examples
labs = np.array([0,1,2,3,4,5,6,7,8,9])
for (image,label) in test_ds.take(5):
    image = tf.expand_dims(image, 0)
    output = model.predict(image)
    out_set = output[0] > qhat
    out_set = labs[out_set]
    print("True output = ",label, " Set is = ",out_set)

#Testing for coverage can be faster with batching methods (DO THIS)
incorrect = 0
correct = 0
for (image,label) in test_ds.take(500):
    image = tf.expand_dims(image, 0)
    output = model.predict(image)
    out_set = output[0] > qhat
    out_set = labs[out_set]
    if label in out_set:
        correct += 1
    else:
        incorrect += 1

print("coverage = ", (correct/(correct+incorrect))*100,"%")




#Adaptive set sizes are better.
scores = []
batch_size = 128
for (image,label) in val_ds.batch(batch_size):
    output = model.predict(image)
    for i in range(tf.size(label)):
        #sort the softmax scores and return the arguments
        s=0
        l = label[i]
        o = output[i]
        while len(o) !=  0:
            m = max(o)
            mi = np.argmax(o)
            s += m
            o = np.delete(o,mi)
            if mi == l:
                break
        scores.append(s)

qhat = np.percentile(scores,(((n+1)*(1-alpha))/n)*100)
print("Adaptive qhat = ",qhat)
plt.hist(scores)
plt.vlines(qhat,0,200, color='r')
plt.savefig("hist_class_adaptive")
plt.close()

#examples
labs = np.array([0,1,2,3,4,5,6,7,8,9])
for (image,label) in test_ds.take(5):
    image = tf.expand_dims(image, 0)
    output = np.squeeze(model.predict(image))
    o = output
    c = 0
    i = 0
    while len(o) != 0:
        m = max(o)
        mi = np.argmax(o)
        c += m
        i += 1
        o = np.delete(o,mi)
        if c > qhat:
            break
    ind = np.argpartition(output,-i)[-i:]
    out_set = labs[ind]
    print("True output = ",label, " Set is = ",out_set)
        
#coverage
correct = 0
incorrect = 0
for (image,label) in test_ds.take(200):
    image = tf.expand_dims(image, 0)
    output = np.squeeze(model.predict(image))
    o = output
    c = 0
    i = 0
    while len(o) != 0:
        m = max(o)
        mi = np.argmax(o)
        c += m
        i += 1
        o = np.delete(o,mi)
        if c > qhat:
            break
    ind = np.argpartition(output,-i)[-i:]
    out_set = labs[ind]
    if label in out_set:
        correct += 1
    else:
        incorrect += 1

print("coverage = ", (correct/(correct+incorrect))*100,"%")
    




