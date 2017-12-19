import tensorflow as tf

def inference(x, xsize, ysize, W_vals=0, b_vals=0):
    '''
    This is a general-purpose softmax inference layer implementation.
    '''
    W_init = tf.constant_initializer(value=W_vals)
    b_init = tf.constant_initializer(value=b_vals)
    W = tf.get_variable('W', [xsize, ysize], initializer=W_init)
    b = tf.get_variable('b', [ysize],        initializer=b_init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    return output
