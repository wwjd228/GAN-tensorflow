import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil


img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = False
to_restore = False
output_path = "output"

max_epoch = 500

patch_size = 5
in_depth = 256
fc1_size = 1024
fc2_size = in_depth*(img_size // 16)
h1_depth = 128

h1_size = 64
h2_size = 128
z_size = 100
batch_size = 128
channel = 1

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def relu(x, alpha=0., max_value=None):
    '''Rectified linear unit
    # Arguments
    alpha: slope of negative section.
    max_value: saturation threshold.
    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x

def add_fclayer( input_size, output_size, data_in, w_name, b_name ):
    fc_w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name=w_name, dtype=tf.float32)
    fc_b = tf.Variable(tf.zeros([output_size]), name=b_name, dtype=tf.float32)
    fc = tf.matmul(data_in, fc_w) + fc_b
    # bn
    fc_scale = tf.Variable(tf.ones([output_size]))
    fc_shift = tf.Variable(tf.zeros([output_size]))
    fc_mean, fc_var = tf.nn.moments(fc, axes=[0])
    epsilon = 0.001
    fc = tf.nn.batch_normalization(fc, fc_mean, fc_var, fc_shift, fc_scale, epsilon)
    fc = relu(fc)
    return fc_w, fc_b, fc

def add_deconvlayer( input_size, output_size, data_in, w_name, b_name ):
    w = tf.Variable(tf.truncated_normal([patch_size, patch_size, input_size, output_size], stddev=0.1), name=w_name, dtype=tf.float32)
    b = tf.Variable(tf.zeros([output_size]), name=b_name, dtype=tf.float32)
    # up sampling with nearsest_neighbor
    shape = data_in.get_shape()		
    size = [2 * int(s) for s in shape[1:3]]
    data_in = tf.image.resize_nearest_neighbor(data_in, size)
    w_T = tf.transpose(w, perm=[0, 1, 3, 2])
    output_shape = [batch_size, int(data_in.get_shape()[1]), int(data_in.get_shape()[2]), output_size]
    deconv = (tf.nn.conv2d_transpose(data_in, w_T, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + b)
    gcnn_mean, gcnn_var = tf.nn.moments(deconv, axes=[0, 1, 2])
    gscale = tf.Variable(tf.ones([output_size]))
    gshift = tf.Variable(tf.zeros([output_size]))  
    epsilon = 0.001
    deconv = tf.nn.batch_normalization(deconv, gcnn_mean, gcnn_var, gshift, gscale, epsilon)
    deconv = relu(deconv)
    return w, b, deconv

def add_convlayer( input_size, output_size, data_in, w_name, b_name, keep_prob, s=1 ):
    w = tf.Variable(tf.truncated_normal([patch_size, patch_size, input_size, output_size], stddev=0.1), name=w_name, dtype=tf.float32)
    b = tf.Variable(tf.zeros([output_size]), name=b_name, dtype=tf.float32)
    conv = (tf.nn.conv2d(data_in, w, strides=[1, s, s, 1], padding='SAME') + b)
    dcnn_mean, dcnn_var = tf.nn.moments(conv, axes=[0, 1, 2])
    dscale = tf.Variable(tf.ones([output_size]))
    dshift = tf.Variable(tf.zeros([output_size]))  
    epsilon = 0.001
    conv = tf.nn.batch_normalization(conv, dcnn_mean, dcnn_var, dshift, dscale, epsilon)
    conv = tf.nn.dropout(relu(conv, alpha=0.2), keep_prob)
    return w, b, conv

def build_generator(z_prior):
    with tf.device('/gpu:0'):
        fcinput_sizes = [z_size, fc1_size]
        fcoutput_sizes = [fc1_size, fc2_size]
        data_in = z_prior
        w_name = "gfc_w"
        b_name = "gfc_b"
        g_params = []
        for fc in range(2):
            w, b, data_in = add_fclayer(fcinput_sizes[fc], fcoutput_sizes[fc], data_in, w_name+str(fc), b_name+str(fc))
            g_params.append(w)
            g_params.append(b)

        data_in = tf.reshape(data_in, [batch_size, img_height//4, img_width//4, in_depth])
        w, b, data_in = add_deconvlayer(in_depth, h1_depth, data_in, "g_w1", "g_b1")
        g_params.append(w)
        g_params.append(b)

        w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, h1_depth, channel], stddev=0.1), name="g_w2", dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([1]), name="g_b2", dtype=tf.float32)
        # up sampling with nearsest_neighbor
        shape = data_in.get_shape()
        print("h2 : ", shape)
        size = [2 * int(s) for s in shape[1:3]]
        data_in = tf.image.resize_nearest_neighbor(data_in, size)
        w2_T = tf.transpose(w2, perm=[0, 1, 3, 2])
        output_shape = [batch_size, int(data_in.get_shape()[1]), int(data_in.get_shape()[2]), channel]
        h2 = (tf.nn.conv2d_transpose(data_in, w2_T, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME') + b2)
        x_generate = tf.nn.tanh(h2)     
        x_generate = tf.reshape(x_generate, [batch_size, img_size])		
        g_params.append(w2)
        g_params.append(b2)
        return x_generate, g_params


def build_discriminator(x_data, x_generated, keep_prob):
    with tf.device('/gpu:0'):       
        x_in = tf.concat(0, [x_data, x_generated])
        x_in = tf.reshape(x_in, [batch_size*2, img_height, img_width, channel])
        w_name = "d_w"
        b_name = "d_b"
        input_sizes = [channel, h1_size]
        output_sizes = [h1_size, h2_size]
        d_params = []
        for c in range(2):
            w, b, x_in = add_convlayer(input_sizes[c], output_sizes[c], x_in, w_name+str(c), b_name+str(c), keep_prob, s=2)
            d_params.append(w)
            d_params.append(b)

        w3 = tf.Variable(tf.truncated_normal([6272, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
        shape = x_in.get_shape().as_list()
        h3 = tf.matmul(tf.reshape(x_in, [batch_size*2, 6272]), w3) + b3
        dscale3 = tf.Variable(tf.ones([1]))
        dshift3 = tf.Variable(tf.zeros([1]))
        dfc_mean3, dfc_var3 = tf.nn.moments(h3, axes=[0]) 
        epsilon = 0.001
        h3 = tf.nn.batch_normalization(h3, dfc_mean3, dfc_var3, dshift3, dscale3, epsilon)
		
        data_logistic = tf.slice(h3, [0, 0], [batch_size, -1], name=None)
        y_data = tf.nn.sigmoid(data_logistic)
        generated_logistic = tf.slice(h3, [batch_size, 0], [-1, -1], name=None)
        y_generated = tf.nn.sigmoid(generated_logistic)
        d_params.append(w3)
        d_params.append(b3)
        return y_data, data_logistic, y_generated, generated_logistic, d_params

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5 # [-1, 1] back to [0, 1]
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    x_generated, g_params = build_generator(z_prior)
    y_data, data_logists, y_generated, generated_logists, d_params = build_discriminator(x_data, x_generated, keep_prob)

    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

#    print(g_params)
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)


    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    for i in range(sess.run(global_step), max_epoch):
        for j in range(30000 // batch_size):
            if i % 25 == 0 and j == 0:
                print("epoch:%s, iter:%s" % (i, j))

            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1 # normalize to -1 ~ 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            sess.run(d_trainer,
                      feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            sess.run(g_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
