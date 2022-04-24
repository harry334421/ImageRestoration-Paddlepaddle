import numpy as np
from numpy.lib.function_base import place
import paddle
from paddle import device
import paddle.fluid as fluid
import time
from paddle.framework import dtype
import paddle.nn as nn
import os
from .base_color import *
from utils.trainable_layers import NNEncLayer, Rebalance_Op
import pdb


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2D):
        super(ECCVGenerator, self).__init__()

        model1 = [
            nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            nn.Conv2D(64, 64, kernel_size=3, stride=2, padding=1),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            nn.Conv2D(128, 128, kernel_size=3, stride=2, padding=1),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=2, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2D(256, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            norm_layer(512),
        ]

        model8 = [nn.Conv2DTranspose(512, 256, kernel_size=4, stride=2, padding=1)]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model8 += [
            nn.ReLU(True),
        ]

        model8 += [
            nn.Conv2D(256, 313, kernel_size=1, stride=1, padding=0),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(axis=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.rebalance_factor = np.load(
            "6364_MechineLearning/Final/model/pretrain/prior_probs.npy"
        )
        # rebalance_factor (313, )
        self.points = np.load(
            "6364_MechineLearning/Final/model/pretrain/pts_in_hull.npy"
        )
        self.points = paddle.to_tensor(self.points, dtype=paddle.float32) / 110.0
        # points (313,2)
        self.rebalance_factor = paddle.to_tensor(
            self.rebalance_factor, dtype=paddle.float32
        )
        self.gt_encoder = NNEncLayer(self.points)
        self.model_out = nn.Conv2D(
            313,
            2,
            kernel_size=1,
            padding=0,
            dilation=1,
            stride=1,
            bias_attr=False,
            # weight_attr=nn.initializer.Assign(paddle.reshape(self.points,(313,2,1,1))
        )
        self.pool = nn.AvgPool2D(4, 4)
        self.channel_size = 313

    # for demo
    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

    # for train
    # def forward(self, input_l, target, if_train=True):
    #     gen = self.model1(self.normalize_l(input_l))
    #     gen = self.model2(gen)
    #     gen = self.model3(gen)
    #     gen = self.model4(gen)
    #     gen = self.model5(gen)
    #     gen = self.model6(gen)
    #     gen = self.model7(gen)
    #     gen = self.model8(gen)  # B*313*H/4*W/4

    #     if if_train:
    #         target = self.pool(self.normalize_ab(target))

    #         target = self.gt_encoder(target)
    #         weight = paddle.argmax(target, axis=-1)
    #         weight = nn.functional.one_hot(weight, self.channel_size)
    #         weight = paddle.matmul(weight, self.rebalance_factor.unsqueeze(1))
    #         # target = Rebalance_Op.apply(target,weight)
    #         gen_reshape = paddle.reshape(paddle.transpose(gen, (0, 2, 3, 1)), (-1, 313))
    #         # B*H*W,313
    #         gen_reshape = Rebalance_Op.apply(gen_reshape, weight)
    #     else:

    #         gen_reshape = self.unnormalize_ab(
    #             self.upsample4(self.model_out(self.softmax(gen)))
    #         )

    #     return gen_reshape, target


def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        model.set_state_dict(
            paddle.load(
                "6364_MechineLearning/Final/model/pretrain/paddle_eccv16.pdparams"
            )
        )
        print("Successfully add model pretrained parameters!")
    return model
