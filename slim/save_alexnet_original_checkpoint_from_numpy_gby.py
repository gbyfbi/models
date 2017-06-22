import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import alexnet_original

numpy_checkpoint_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet_faster_r_cnn.npy'
output_checkpoint_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/checkpoints/alexnet_original_imagenet'
output_checkpoint_file_path = output_checkpoint_dir+'/'+'alexnet_original'
image_size = alexnet_original.alexnet_original.default_image_size
num_classes = 1000


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    fake_image = tf.zeros((32, 227, 227, 3), name='input_image')
    with slim.arg_scope(alexnet_original.alexnet_original_arg_scope()):
        logits, end_points = alexnet_original.alexnet_original(fake_image, num_classes=num_classes, is_training=False)

    if numpy_checkpoint_path.endswith('.npy'):
        var_init_op_list = []
        import numpy as np
        data_dict = np.load(numpy_checkpoint_path).item()
        for key in sorted(data_dict):
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        var_init_op = var.assign(data_dict[key][subkey])
                        var_init_op_list.append(var_init_op)
                        print("assign pretrain model " + subkey + " to " + key)
                    except ValueError:
                        print("ignore " + key + '/' + subkey)
                        raise
        from tensorflow.python.ops import control_flow_ops
        numpy_init_op = control_flow_ops.group(*var_init_op_list, name='init_var_from_numpy')
    else:
        raise ValueError("checkpoint path is not ended with .npy!")
    # init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

    with tf.Session() as sess:
        # init_fn(sess)
        sess.run(numpy_init_op)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)) # MODEL_VARIABLES is just for contrib layers or the ones I defined:)
        saver.save(sess, output_checkpoint_file_path)
        writer = tf.summary.FileWriter(output_checkpoint_dir, sess.graph)
        writer.close()


