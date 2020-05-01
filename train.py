# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse

from model.unet import UNet



import easydict

args = easydict.EasyDict({

    "experiment_dir": 'C:/Users/1-16/python/neural-fonts-master/experiment',

    "experiment_id": 0,

    "image_size": 128,

    "L1_penalty": 100,

    "Lconst_penalty": 15,

    "Ltv_penalty": 0.0,

    "Lcategory_penalty": 1.0,

    "embedding_num": 40,

    "embedding_dim": 128,
    "batch_size": 100,
    "lr": 0.001,

    "schedule": 10,
    "resume": 1,
    "freeze_encoder": 0,
    "fine_tune": None,
    "inst_norm": 0,
    "sample_steps": 10,
    "checkpoint_steps": 500,
    "flip_labels": None,
    "no_val": None

})
'''
def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty)
        model.register_session(sess)
        if args.flip_labels:
            model.build_model(is_training=True, inst_norm=args.inst_norm, no_target_source=True)
        else:
            model.build_model(is_training=True, inst_norm=args.inst_norm)
        fine_tune_list = None
        if args.fine_tune:
            ids = args.fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,
                    flip_labels=args.flip_labels, no_val=args.no_val)
'''

def main():
    model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                 input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                 embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                 Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty)

    if args.flip_labels:
        model.build_model(is_training=True, inst_norm=args.inst_norm, no_target_source=True)
    else:
        model.build_model(is_training=True, inst_norm=args.inst_norm)
    fine_tune_list = None
    if args.fine_tune:
        ids = args.fine_tune.split(",")
        fine_tune_list = set([int(i) for i in ids])
    model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,
                flip_labels=args.flip_labels, no_val=args.no_val)


if __name__ == '__main__':
    main()





