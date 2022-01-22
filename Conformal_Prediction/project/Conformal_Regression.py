from distutils.command.build import build
from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa

print("Setup Complete")

#Goal-
#- Provide a range of values that we can say are 90% certain will contain any
#  future values within the bounds.

#Read the insurance data and plot insurance bmi vs charges
raw_data = pd.read_csv("insurance.csv")
print(raw_data.head())
bmi = raw_data["bmi"]
charge = raw_data["charges"]

plt.scatter(bmi,charge)
plt.xlabel("BMI")
plt.ylabel("Insurance Charge")
plt.savefig("bmi_vs_charge.png")
plt.close()


#Split the data into training and calibration(testing) at 70:30 randomly
data = pd.concat([raw_data["bmi"], raw_data["charges"]], axis=1)
num_rows = len(raw_data.index)
train_data = data.sample(frac=0.7)
test_data = data.drop(train_data.index)

print("train data points = ",len(train_data.index))
print("test data points = ", len(test_data.index))

#Build the NN model (loss used in MAE)
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#Setup Normalizer and scale values between 0 and 1 for easier regression
normalizer = layers.Normalization(input_shape=[1,], axis=None)
normalizer.adapt(np.array(train_data['bmi'])) #scaling happens here

#Build the simple NN model
my_model = build_and_compile_model(normalizer)

#Train the model and split the train data into train and validation sets
history = my_model.fit(
    train_data['bmi'],
    train_data['charges'],
    validation_split=0.2,
    verbose=0, epochs=50
)

def plot_loss(history,name):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 15000])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.savefig(name)
  plt.close()

def plot_data(x, y,name,train_data):
  plt.scatter(train_data['bmi'], train_data['charges'], label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('BMI')
  plt.xlim([15,60])
  plt.ylabel('Charge')
  plt.legend()
  plt.savefig(name)
  plt.close()

plot_loss(history,"plot_loss_dnn")

#Plot the regressed mean 
x=tf.linspace(0.0,raw_data["bmi"].max(),raw_data["bmi"].max()+1)
y=my_model.predict(x)
plot_data(x,y,"plot_output_dnn",train_data)

#Conformal fixed bounds
#calculate the residuals between the model and the test data set
alpha = 0.1 # 1-alpha is the confidence wanted from the model i.e 90% here
predictions = np.squeeze(my_model.predict(test_data['bmi']))
y = test_data['charges'].to_numpy()
residuals = predictions - y
residuals = np.absolute(residuals)

#Calculate the 1-alpha percentile 
q_fixed = np.percentile(residuals,(1-alpha)*100)

#Plot histogram of the residuals in the test set
plt.hist(residuals)
plt.vlines(q_fixed,0,200, color='r')
plt.savefig("hist_q_fixed")
plt.close()

#add and subtract the quantiles from the model
x = tf.linspace(0.0,raw_data["bmi"].max(),raw_data["bmi"].max()+1)
y=my_model.predict(x)
upper_conf = y + q_fixed
lower_conf = y - q_fixed

plt.scatter(train_data['bmi'], train_data['charges'], label='Train Data',alpha=0.5)
plt.scatter(test_data['bmi'], test_data['charges'], label='Test Data',alpha=0.5)
plt.plot(x, y, color='k', label='Regression')
plt.plot(x, upper_conf, color='b', label='90% Coverage')
plt.plot(x, lower_conf, color='b', label='90% Coverage')
plt.xlabel('BMI')
plt.ylabel('Charge')
plt.xlim([15,60])
plt.legend()
plt.savefig("fixed_bounds_plot")
plt.close()

#check coverage basic but not the best as we use this train test split already
s = 0
for r in residuals:
    if r > q_fixed:
        s+=1

test_coverage_fixed = s/len(test_data['bmi'].index)
print("Test coverage = ",test_coverage_fixed)


#showing training residuals (dont use for analysis)
training_predictions = np.squeeze(my_model.predict(train_data['bmi']))
train_y = train_data['charges'].to_numpy()
train_residuals = training_predictions - train_y
train_residuals = np.absolute(train_residuals)

s = 0
for r in train_residuals:
    if r > q_fixed:
        s+=1
train_coverage_fixed = s/len(train_data['bmi'].index)
print("Training coverage = ",train_coverage_fixed)

#Adaptive Regression with pinball loss
#https://arxiv.org/pdf/1905.03222.pdf

alpha = 0.1
n = len(test_data.index)

#build a NN with pinball loss 
def build_and_compile_pinball_model(norm,t):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss=tfa.losses.PinballLoss(tau=t),
        optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#Train the upper confidence bound model(Possible to train simultaniously)
pinball_upper_model = build_and_compile_pinball_model(normalizer,1-alpha/2)
upper_history = pinball_upper_model.fit(
    train_data['bmi'],
    train_data['charges'],
    validation_split=0.2,
    verbose=0, epochs=50
)

#Train the lower confidence bound model
pinball_lower_model = build_and_compile_pinball_model(normalizer,alpha/2)
lower_history = pinball_lower_model.fit(
    train_data['bmi'],
    train_data['charges'],
    validation_split=0.2,
    verbose=0, epochs=50
)

plt.plot(upper_history.history['loss'], label='upper_loss')
plt.plot(upper_history.history['val_loss'], label='upper_val_loss')
plt.plot(lower_history.history['loss'], label='lower_loss')
plt.plot(lower_history.history['val_loss'], label='lower_val_loss')
plt.ylim([100, 15000])
plt.yscale("log")
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig("pinball_loss")
plt.close()


#Conformal inference
#create the model lines
x=tf.linspace(0.0,raw_data["bmi"].max(),raw_data["bmi"].max()+1)
y_upper=pinball_upper_model.predict(x)
y_lower=pinball_lower_model.predict(x)

y_upper_pred=np.squeeze(pinball_upper_model.predict(test_data['bmi']))
y_lower_pred=np.squeeze(pinball_lower_model.predict(test_data['bmi']))

#Score negative if test point is between the 2 models and positive if outside the bounds
#This allows the model to shrink as well as grow if original model is too forgiving
scores = np.max([y_lower_pred - test_data['charges'],test_data['charges'] - y_upper_pred],axis=0)

#the percentile calculated here is corrected for finite sampling
q_pinball = np.percentile(scores,(((n+1)*(1-alpha))/n)*100)
print("qhat_pinball = ", q_pinball)

plt.hist(scores)
plt.vlines(q_pinball,0,200)
plt.savefig("hist_q_pinball")
plt.close()

plt.scatter(train_data['bmi'], train_data['charges'], label='Train Data',alpha=0.5)
plt.scatter(test_data['bmi'], test_data['charges'], label='Test Data',alpha=0.5)
plt.plot(x, y_upper, color='k', label='45% pinball loss')
plt.plot(x, y_lower, color='k', label='5% pinball loss')
plt.plot(x, y_upper+q_pinball, color='r', label="90% coverage")
plt.plot(x, y_lower-q_pinball, color='r')
plt.xlabel('BMI')
plt.xlim([15,60])
plt.ylabel('Charge')
plt.legend()
plt.savefig("pinball_data")
plt.close()

#check coverage again not ideal way to do this
s=0
scores = np.max([y_lower_pred - test_data['charges'],test_data['charges'] - y_upper_pred],axis=0)
for sc in scores:
    if sc > q_pinball:
        s+=1
test_coverage_pin = s/len(test_data['bmi'].index)
print("Test coverage = ",test_coverage_pin)

s = 0
y_upper_pred=np.squeeze(pinball_upper_model.predict(train_data['bmi']))
y_lower_pred=np.squeeze(pinball_lower_model.predict(train_data['bmi']))

scores = np.max([y_lower_pred - train_data['charges'],train_data['charges'] - y_upper_pred],axis=0)
for sc in scores:
    if sc > q_pinball:
        s+=1

train_coverage_pin = s/len(train_data['bmi'].index)
print("Train coverage = ",train_coverage_pin)

#print outputs to file
more_lines = [test_coverage_fixed, train_coverage_fixed, test_coverage_pin,train_coverage_pin]
with open('Reg_coverages.txt', 'a') as f:
    f.write('\n')
    f.write(str(more_lines))
