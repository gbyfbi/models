import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import alexnet_original

numpy_checkpoint_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet_faster_r_cnn.npy'
output_checkpoint_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/checkpoints/alexnet_original_imagenet'
# output_checkpoint_file_path = output_checkpoint_dir+'/'+'alexnet_original'
num_classes = 1000
output_checkpoint_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/flowers-models/alexnet_original_678_no_lrn_sub_RGB_mean_ckpt_convert_from_npy'
# output_checkpoint_file_path = output_checkpoint_dir+'/'+'model'
image_size = alexnet_original.alexnet_original.default_image_size
num_classes = 5


with tf.Graph().as_default():
    with tf.device('/cpu:0'):
        tf.logging.set_verbosity(tf.logging.INFO)
        fake_image = tf.zeros((32, 227, 227, 3), name='input_image')
        with slim.arg_scope(alexnet_original.alexnet_original_arg_scope()):
            logits, end_points = alexnet_original.alexnet_original(fake_image, num_classes=num_classes, is_training=False)
            # with tf.Session(config=tf.ConfigProto()) as sess:
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)) # MODEL_VARIABLES is just for contrib layers or the ones I defined:)
                latest_checkpoint_path = tf.train.latest_checkpoint(output_checkpoint_dir)
                saver.restore(sess, latest_checkpoint_path)
                with tf.variable_scope('fc8', reuse=True) as vs:
                    var = tf.get_variable('weights')
                    var_np = sess.run(var)
                    print(var_np.shape)
                    print(var_np)
            # saver.save(sess, output_checkpoint_file_path)


