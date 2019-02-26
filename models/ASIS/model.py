import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from loss import *

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, 3:]
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points_sem = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='sem_fa_layer1')
    l2_points_sem = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_sem, [256,256], is_training, bn_decay, scope='sem_fa_layer2')
    l1_points_sem = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_sem, [256,128], is_training, bn_decay, scope='sem_fa_layer3')
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128,128,128], is_training, bn_decay, scope='sem_fa_layer4')

    # FC layers
    net_sem = tf_util.conv1d(l0_points_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc1', bn_decay=bn_decay)
    net_sem_cache = tf_util.conv1d(net_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_cache',  bn_decay=bn_decay)  

    # ins
    l3_points_ins = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='ins_fa_layer1')
    l2_points_ins = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_ins, [256,256], is_training, bn_decay, scope='ins_fa_layer2')
    l1_points_ins = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_ins, [256,128], is_training, bn_decay, scope='ins_fa_layer3')
    l0_points_ins = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_ins, [128,128,128], is_training, bn_decay, scope='ins_fa_layer4')

    net_ins = tf_util.conv1d(l0_points_ins, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc1', bn_decay=bn_decay)

    net_ins = net_ins + net_sem_cache
    net_ins = tf_util.dropout(net_ins, keep_prob=0.5, is_training=is_training, scope='ins_dp1')
    net_ins = tf_util.conv1d(net_ins, 5, 1, padding='VALID', activation_fn=None, scope='ins_fc4')

    k = 30
    adj_matrix = tf_util.pairwise_distance_l1(net_ins)
    nn_idx = tf_util.knn_thres(adj_matrix, k=k)
    nn_idx = tf.stop_gradient(nn_idx)

    net_sem = tf_util.get_local_feature(net_sem, nn_idx=nn_idx, k=k)# [b, n, k, c]
    net_sem = tf.reduce_max(net_sem, axis=-2, keep_dims=False)

    net_sem = tf_util.dropout(net_sem, keep_prob=0.5, is_training=is_training, scope='sem_dp1')
    net_sem = tf_util.conv1d(net_sem, num_class, 1, padding='VALID', activation_fn=None, scope='sem_fc4')

    
    return net_sem, net_ins


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.001
    
    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, feature_dim,
                                         delta_v, delta_d, param_var, param_dist, param_reg)

    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist, l_reg

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
