import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(BASE_DIR) 
sys.path.append(os.path.join(BASE_DIR, 'utils')) 


def discriminative_loss_single(prediction, correct_label, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    #correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])

    ### Count instances
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)

    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)
    

    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    distance = tf.norm(tmp_distance, ord=1, axis=1)

    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # Filter out zeros from same cluster subtraction
    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
    mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

    #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    #zero_vector = tf.zeros(1, dtype=tf.float32)
    #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    def rt_0(): return 0.
    def rt_l_dist(): return l_dist
    l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)
    
    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,
                                                                     delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                           prediction,
                                                                                           output_ta_loss,
                                                                                           output_ta_var,
                                                                                           output_ta_dist,
                                                                                           output_ta_reg,
                                                                                           0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_single_multicate(sem_label, prediction, correct_label, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param sem_label: semantic label
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''
    unique_sem_label, unique_id, counts = tf.unique_with_counts(sem_label)
    num_sems = tf.size(unique_sem_label)

    def cond(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        return tf.less(i, num_sems)

    def body(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        inds = tf.equal(i, unique_id)
        cur_pred = tf.boolean_mask(prediction, inds)
        cur_label = tf.boolean_mask(correct_label, inds)
        cur_discr_loss, cur_l_var, cur_l_dist, cur_l_reg = discriminative_loss_single(cur_pred, cur_label, feature_dim,
                            delta_v, delta_d, param_var, param_dist, param_reg)
        out_loss = out_loss.write(i, cur_discr_loss)
        out_var = out_var.write(i, cur_l_var)
        out_dist = out_dist.write(i, cur_l_dist)
        out_reg = out_reg.write(i, cur_l_reg)

        return i + 1, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg

    output_ta_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    loop = [0, num_sems, unique_id, prediction, correct_label, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg]
    _, _, _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op = tf.while_loop(cond, body, loop)

    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_sum(out_loss_op)
    l_var = tf.reduce_sum(out_var_op)
    l_dist = tf.reduce_sum(out_dist_op)
    l_reg = tf.reduce_sum(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_multicate(sem_label, prediction, correct_label, feature_dim,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss for multiple categories.
    :return: discriminative loss and its three components
    '''

    def cond(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single_multicate(sem_label[i], prediction[i], correct_label[i], feature_dim,
                                                                     delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return sem, label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [sem_label,
                                                                                           correct_label,
                                                                                           prediction,
                                                                                           output_ta_loss,
                                                                                           output_ta_var,
                                                                                           output_ta_dist,
                                                                                           output_ta_reg,
                                                                                           0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)


    return disc_loss, l_var, l_dist, l_reg
