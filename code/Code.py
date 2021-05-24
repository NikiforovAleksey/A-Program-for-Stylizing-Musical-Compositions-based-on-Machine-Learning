!pip install --upgrade librosa==0.7.2
!pip install numba==0.48

base_music = 'Raffaele Rinciari - Acqua.mp3' 
style_music = 'organnaya-muzyka-bez-nazvaniya_(mp3IQ.net).mp3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import math
import IPython.display as ipd

result_prefix = "output"

total_variation_weight = 1e-7#1e-6
style_weight = 1e-6 #1e-6
content_weight = 1e-7 #2.5e-8
style_content_weight = 1e-8

img_nrows = 513 #257
img_ncols = 1000

start_col = 0
finish_col = start_col+img_ncols

data_type = np.float32

def my_amplitude_to_db(X, amin = 1e-6, amax = 1e+4):
    X_out = np.zeros(X.shape, dtype = np.float64)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_out[i,j] = math.log10(min(amax, max(amin, X[i,j].real**2 + X[i,j].imag**2)))
    return X_out

def my_db_to_amplitude(X):
    X_out = np.zeros(X.shape, dtype = np.float64)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_out[i,j] = math.sqrt(10.0**X[i,j])
    return X_out


def wav_to_np(filename):
    audio, sample_rate = librosa.load(filename)
    X = librosa.stft(audio, n_fft = (img_nrows-1)*2)
    Xdb = my_amplitude_to_db(X[:,start_col:finish_col])   #librosa.amplitude_to_db
    X_arr = np.zeros((Xdb.shape[0], Xdb.shape[1], 3), dtype=np.uint8)
    e_max = Xdb.max()
    e_min = Xdb.min()
    e_d = e_max-e_min
    for i in range(Xdb.shape[0]):
        for j in range(Xdb.shape[1]):
            a = int((Xdb[i,j]-e_min)/e_d * 255)
            X_arr[i,j,:] = [a, a, a]
    return X_arr

def preprocess_image(img):
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=data_type)


def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

model = vgg19.VGG19(weights="imagenet", include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = [
    #"block5_conv2",
    "block4_conv2",
    "block3_conv2",
    "block2_conv2",
    "block1_conv2"
]  
content_layer_names = [
    "block5_conv2",
    "block4_conv2",
    "block3_conv2",
]

style_content_layer_names = [
    #"block5_conv2",
    #"block4_conv2",
    "block3_conv2",
    "block2_conv2",
    "block1_conv2"
]


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    loss = tf.zeros(shape=(), dtype=data_type)

    for layer_name in content_layer_names:
        layer_features = features[layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        cl = content_loss(base_image_features, combination_features)
        loss += (content_weight/len(content_layer_names)) * tf.cast(cl, data_type)

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * tf.cast(sl, data_type)

    for layer_name in style_content_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        scl = content_loss(style_reference_features, combination_features)
        loss += (style_content_weight/len(style_content_layer_names)) * tf.cast(scl, data_type)

    loss += total_variation_weight * tf.cast(total_variation_loss(combination_image), data_type)
    return loss

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

from IPython.display import Image, display

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=200.0, decay_steps=200, decay_rate=0.96
    )
)

base_image = preprocess_image(wav_to_np(base_music))
style_reference_image = preprocess_image(wav_to_np(style_music))
combination_image = tf.Variable(preprocess_image(wav_to_np(base_music)))

from IPython.display import Image, display
img1 = deprocess_image(base_image.numpy())
img2 = deprocess_image(style_reference_image.numpy())
fname1 = 'base_show.png'
fname2 = 'style_show.png'
keras.preprocessing.image.save_img(fname1, img1)
keras.preprocessing.image.save_img(fname2, img2)
display(Image(fname1))
display(Image(fname2))

def compute_all_loses(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    loss_c = 0
    for layer_name in content_layer_names:
        layer_features = features[layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        cl = content_loss(base_image_features, combination_features)
        loss_c += ((content_weight/len(content_layer_names)) * tf.cast(cl, data_type)).numpy()

    loss_s = 0
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss_s += ((style_weight / len(style_layer_names)) * tf.cast(sl, data_type)).numpy()

    loss_sc = 0
    for layer_name in style_content_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        scl = content_loss(style_reference_features, combination_features)
        loss_sc += ((style_content_weight/len(style_content_layer_names)) * tf.cast(scl, data_type)).numpy()

    loss_v = (total_variation_weight * tf.cast(total_variation_loss(combination_image), data_type)).numpy()
    return loss_c, loss_s, loss_sc, loss_v


from datetime import datetime
ls = compute_all_loses(combination_image, base_image, style_reference_image)
print(ls)
iterations = 4000
print(datetime.now())
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    #print(i)
    if i <= 5:
        print("Iteration %d: loss=%.2f" % (i, loss))
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        ls = compute_all_loses(combination_image, base_image, style_reference_image)
        print(ls)
        if i % 1000 == 0:
            np.save(result_prefix + "_new_at_iteration_%d.npy" % i, combination_image.numpy())
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix + "_at_iteration_%d.png" % i
            keras.preprocessing.image.save_img(fname, img)
            display(Image(fname))


def img_to_music(img_arr, filename):
    img_arr = deprocess_image(img_arr)
    audio, sample_rate = librosa.load(base_music)
    X = librosa.stft(audio, n_fft = (img_nrows-1)*2)[:,start_col:finish_col]
    X_l = my_amplitude_to_db(np.abs(X))   #librosa.amplitude_to_db
    e_max = X_l.max()
    e_min = X_l.min()
    e_d = e_max-e_min
    out_arr = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    out_arr_r = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    out_arr_i = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            out_arr[i,j] = 1.0*img_arr[i,j,:].mean()*e_d / 255 + e_min
    out_arr = my_db_to_amplitude(out_arr)      #librosa.db_to_amplitude
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            out_arr_r[i,j] = out_arr[i,j]*math.cos(np.angle(X[i,j]))
            out_arr_i[i,j] = out_arr[i,j]*math.sin(np.angle(X[i,j]))
    X.real = out_arr_r
    X.imag = out_arr_i
    reconstructed_audio = librosa.istft(X)
    librosa.output.write_wav(filename, reconstructed_audio, sample_rate)


img_to_music(X_test, 'test_test.wav')
ipd.Audio('test_test.wav')
