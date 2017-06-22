import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import inception_v4
import os
output_checkpoint_dir = '/tmp/tf_debug'
if not os.path.exists(output_checkpoint_dir):
    os.makedirs(output_checkpoint_dir)
output_checkpoint_file_path = output_checkpoint_dir + '/' + 'inception_v4'
checkpoint_file_path = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/checkpoints/inception_v4.ckpt'
image_size = inception_v4.inception_v4.default_image_size
num_classes = 1001


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    fake_image = tf.zeros((32, image_size, image_size, 3), name='input_image')
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, end_points = inception_v4.inception_v4(fake_image, num_classes=num_classes, is_training=False)

    # checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_file_path, slim.get_variables_to_restore())

    with tf.Session() as sess:
        init_fn(sess)
        # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)) # MODEL_VARIABLES is just for contrib layers or the ones I defined:)
        # saver.save(sess, output_checkpoint_file_path)
        writer = tf.summary.FileWriter(output_checkpoint_dir, sess.graph)
        writer.close()


