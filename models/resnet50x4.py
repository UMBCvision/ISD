import torch
import torch.nn as nn
import pdb

class Resnet50_X4(nn.Module) :

    def __init__(self):
        super(Resnet50_X4, self).__init__()
        norm_layer = nn.BatchNorm2d


        self.conv2d = nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.batch_normalization = norm_layer(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # layer 1

        # b1
        self.conv2d_1 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_1 = norm_layer(1024)

        self.conv2d_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_2 = norm_layer(256)

        self.conv2d_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_3 = norm_layer(256)

        self.conv2d_4 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_4 = norm_layer(1024)

        # b2
        self.conv2d_5 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_5 = norm_layer(256)

        self.conv2d_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_6 = norm_layer(256)

        self.conv2d_7 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_7 = norm_layer(1024)

        # b3
        self.conv2d_8 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_8 = norm_layer(256)

        self.conv2d_9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_9 = norm_layer(256)

        self.conv2d_10 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_10 = norm_layer(1024)

        # layer 2

        # b1
        self.conv2d_11 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_11 = norm_layer(2048)

        self.conv2d_12 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_12 = norm_layer(512)

        self.conv2d_13 = nn.Conv2d(512, 512, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_13 = norm_layer(512)

        self.conv2d_14 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_14 = norm_layer(2048)

        # b2
        self.conv2d_15 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_15 = norm_layer(512)

        self.conv2d_16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_16 = norm_layer(512)

        self.conv2d_17 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_17 = norm_layer(2048)

        # b3
        self.conv2d_18 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_18 = norm_layer(512)

        self.conv2d_19 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_19 = norm_layer(512)

        self.conv2d_20 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_20 = norm_layer(2048)

        # b4
        self.conv2d_21 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_21 = norm_layer(512)

        self.conv2d_22 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_22 = norm_layer(512)

        self.conv2d_23 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_23 = norm_layer(2048)

        # layer 3

        # b1
        self.conv2d_24 = nn.Conv2d(2048, 4096, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_24 = norm_layer(4096)

        self.conv2d_25 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_25 = norm_layer(1024)

        self.conv2d_26 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_26 = norm_layer(1024)

        self.conv2d_27 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_27 = norm_layer(4096)

        # b2
        self.conv2d_28 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_28 = norm_layer(1024)

        self.conv2d_29 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False , padding=1)
        self.batch_normalization_29 = norm_layer(1024)

        self.conv2d_30 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_30 = norm_layer(4096)

        # b3
        self.conv2d_31 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_31 = norm_layer(1024)

        self.conv2d_32 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_32 = norm_layer(1024)

        self.conv2d_33 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_33 = norm_layer(4096)

        # b4
        self.conv2d_34 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_34 = norm_layer(1024)

        self.conv2d_35 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_35 = norm_layer(1024)

        self.conv2d_36 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_36 = norm_layer(4096)

        # b5
        self.conv2d_37 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_37 = norm_layer(1024)

        self.conv2d_38 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_38 = norm_layer(1024)

        self.conv2d_39 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_39 = norm_layer(4096)

        # b6
        self.conv2d_40 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_40 = norm_layer(1024)

        self.conv2d_41 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_41 = norm_layer(1024)

        self.conv2d_42 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_42 = norm_layer(4096)

        # layer 4

        # b1
        self.conv2d_43 = nn.Conv2d(4096, 8192, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_43 = norm_layer(8192)

        self.conv2d_44 = nn.Conv2d(4096, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_44 = norm_layer(2048)

        self.conv2d_45 = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_45 = norm_layer(2048)

        self.conv2d_46 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_46 = norm_layer(8192)

        # b2
        self.conv2d_47 = nn.Conv2d(8192, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_47 = norm_layer(2048)

        self.conv2d_48 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_48 = norm_layer(2048)

        self.conv2d_49 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_49 = norm_layer(8192)

        # b2
        self.conv2d_50 = nn.Conv2d(8192, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_50 = norm_layer(2048)

        self.conv2d_51 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_51 = norm_layer(2048)

        self.conv2d_52 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_52 = norm_layer(8192)

        self.fc = nn.Linear(8192 , 1000)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.weight , 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0.2)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # pdb.set_trace()
        x = self.conv2d(x)
        x = self.relu(self.batch_normalization(x))
        x = self.maxpool(x)

        # pdb.set_trace()

        l0 = x

        # layer1
        # b1
        # pdb.set_trace()
        shortcut = self.batch_normalization_1(self.conv2d_1(x))
        x = self.relu(self.batch_normalization_2(self.conv2d_2(x)))
        x = self.relu(self.batch_normalization_3(self.conv2d_3(x)))
        x = self.batch_normalization_4(self.conv2d_4(x))
        x = self.relu(x + shortcut)



        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_5(self.conv2d_5(x)))
        x = self.relu(self.batch_normalization_6(self.conv2d_6(x)))
        x = self.batch_normalization_7(self.conv2d_7(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_8(self.conv2d_8(x)))
        x = self.relu(self.batch_normalization_9(self.conv2d_9(x)))
        x = self.batch_normalization_10(self.conv2d_10(x))
        x = self.relu(x + shortcut)

        # pdb.set_trace()

        l1 = x

        # layer2
        # b1
        shortcut = self.batch_normalization_11(self.conv2d_11(x))
        x = self.relu(self.batch_normalization_12(self.conv2d_12(x)))
        x = self.relu(self.batch_normalization_13(self.conv2d_13(x)))
        x = self.batch_normalization_14(self.conv2d_14(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_15(self.conv2d_15(x)))
        x = self.relu(self.batch_normalization_16(self.conv2d_16(x)))
        x = self.batch_normalization_17(self.conv2d_17(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_18(self.conv2d_18(x)))
        x = self.relu(self.batch_normalization_19(self.conv2d_19(x)))
        x = self.batch_normalization_20(self.conv2d_20(x))
        x = self.relu(x + shortcut)

        # b4
        shortcut = x
        x = self.relu(self.batch_normalization_21(self.conv2d_21(x)))
        x = self.relu(self.batch_normalization_22(self.conv2d_22(x)))
        x = self.batch_normalization_23(self.conv2d_23(x))
        x = self.relu(x + shortcut)

        # pdb.set_trace()

        l2 = x

        # layer3
        # b1
        shortcut = self.batch_normalization_24(self.conv2d_24(x))
        x = self.relu(self.batch_normalization_25(self.conv2d_25(x)))
        x = self.relu(self.batch_normalization_26(self.conv2d_26(x)))
        x = self.batch_normalization_27(self.conv2d_27(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_28(self.conv2d_28(x)))
        x = self.relu(self.batch_normalization_29(self.conv2d_29(x)))
        x = self.batch_normalization_30(self.conv2d_30(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_31(self.conv2d_31(x)))
        x = self.relu(self.batch_normalization_32(self.conv2d_32(x)))
        x = self.batch_normalization_33(self.conv2d_33(x))
        x = self.relu(x + shortcut)

        # b4
        shortcut = x
        x = self.relu(self.batch_normalization_34(self.conv2d_34(x)))
        x = self.relu(self.batch_normalization_35(self.conv2d_35(x)))
        x = self.batch_normalization_36(self.conv2d_36(x))
        x = self.relu(x + shortcut)

        # b5
        shortcut = x
        x = self.relu(self.batch_normalization_37(self.conv2d_37(x)))
        x = self.relu(self.batch_normalization_38(self.conv2d_38(x)))
        x = self.batch_normalization_39(self.conv2d_39(x))
        x = self.relu(x + shortcut)

        # b6
        shortcut = x
        x = self.relu(self.batch_normalization_40(self.conv2d_40(x)))
        x = self.relu(self.batch_normalization_41(self.conv2d_41(x)))
        x = self.batch_normalization_42(self.conv2d_42(x))
        x = self.relu(x + shortcut)

        # pdb.set_trace()

        l3 = x
        # layer4
        # b1
        shortcut = self.batch_normalization_43(self.conv2d_43(x))
        x = self.relu(self.batch_normalization_44(self.conv2d_44(x)))
        x = self.relu(self.batch_normalization_45(self.conv2d_45(x)))
        x = self.batch_normalization_46(self.conv2d_46(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_47(self.conv2d_47(x)))
        x = self.relu(self.batch_normalization_48(self.conv2d_48(x)))
        x = self.batch_normalization_49(self.conv2d_49(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_50(self.conv2d_50(x)))
        x = self.relu(self.batch_normalization_51(self.conv2d_51(x)))
        x = self.batch_normalization_52(self.conv2d_52(x))
        x = self.relu(x + shortcut)

        l4 = x
        # pdb.set_trace()

        # pdb.set_trace()
        x = self.avgpool(x)
        x = torch.flatten(x , 1)

        # pdb.set_trace()
        # x = self.fc(x)

        return x


#  ('base_model/conv2d/kernel', [7, 7, 3, 256]),

#  ('base_model/conv2d_1/kernel', [1, 1, 256, 1024]),
#  ('base_model/conv2d_2/kernel', [1, 1, 256, 256]),
#  ('base_model/conv2d_3/kernel', [3, 3, 256, 256]),
#  ('base_model/conv2d_4/kernel', [1, 1, 256, 1024]),

#  ('base_model/conv2d_5/kernel', [1, 1, 1024, 256]),
#  ('base_model/conv2d_6/kernel', [3, 3, 256, 256]),
#  ('base_model/conv2d_7/kernel', [1, 1, 256, 1024]),

#  ('base_model/conv2d_8/kernel', [1, 1, 1024, 256]),
#  ('base_model/conv2d_9/kernel', [3, 3, 256, 256]),
#  ('base_model/conv2d_10/kernel', [1, 1, 256, 1024]),







#  ('base_model/conv2d_11/kernel', [1, 1, 1024, 2048]),
#  ('base_model/conv2d_12/kernel', [1, 1, 1024, 512]),
#  ('base_model/conv2d_13/kernel', [3, 3, 512, 512]),
#  ('base_model/conv2d_14/kernel', [1, 1, 512, 2048]),
#  ('base_model/conv2d_15/kernel', [1, 1, 2048, 512]),
#  ('base_model/conv2d_16/kernel', [3, 3, 512, 512]),
#  ('base_model/conv2d_17/kernel', [1, 1, 512, 2048]),
#  ('base_model/conv2d_18/kernel', [1, 1, 2048, 512]),
#  ('base_model/conv2d_19/kernel', [3, 3, 512, 512]),
#  ('base_model/conv2d_20/kernel', [1, 1, 512, 2048]),
#  ('base_model/conv2d_21/kernel', [1, 1, 2048, 512]),
#  ('base_model/conv2d_22/kernel', [3, 3, 512, 512]),
#  ('base_model/conv2d_23/kernel', [1, 1, 512, 2048]),



#  ('base_model/conv2d_24/kernel', [1, 1, 2048, 4096]),
#  ('base_model/conv2d_25/kernel', [1, 1, 2048, 1024]),
#  ('base_model/conv2d_26/kernel', [3, 3, 1024, 1024]),
#  ('base_model/conv2d_27/kernel', [1, 1, 1024, 4096]),
#  ('base_model/conv2d_28/kernel', [1, 1, 4096, 1024]),
#  ('base_model/conv2d_29/kernel', [3, 3, 1024, 1024]),

#  ('base_model/conv2d_30/kernel', [1, 1, 1024, 4096]),
#  ('base_model/conv2d_31/kernel', [1, 1, 4096, 1024]),
#  ('base_model/conv2d_32/kernel', [3, 3, 1024, 1024]),
#  ('base_model/conv2d_33/kernel', [1, 1, 1024, 4096]),
#  ('base_model/conv2d_34/kernel', [1, 1, 4096, 1024]),
#  ('base_model/conv2d_35/kernel', [3, 3, 1024, 1024]),
#  ('base_model/conv2d_36/kernel', [1, 1, 1024, 4096]),
#  ('base_model/conv2d_37/kernel', [1, 1, 4096, 1024]),
#  ('base_model/conv2d_38/kernel', [3, 3, 1024, 1024]),
#  ('base_model/conv2d_39/kernel', [1, 1, 1024, 4096]),

#  ('base_model/conv2d_40/kernel', [1, 1, 4096, 1024]),
#  ('base_model/conv2d_41/kernel', [3, 3, 1024, 1024]),
#  ('base_model/conv2d_42/kernel', [1, 1, 1024, 4096]),


#  ('base_model/conv2d_43/kernel', [1, 1, 4096, 8192]),
#  ('base_model/conv2d_44/kernel', [1, 1, 4096, 2048]),
# ('base_model/conv2d_45/kernel', [3, 3, 2048, 2048]),
#  ('base_model/conv2d_46/kernel', [1, 1, 2048, 8192]),
#  ('base_model/conv2d_47/kernel', [1, 1, 8192, 2048]),
#  ('base_model/conv2d_48/kernel', [3, 3, 2048, 2048]),
#  ('base_model/conv2d_49/kernel', [1, 1, 2048, 8192]),

#  ('base_model/conv2d_50/kernel', [1, 1, 8192, 2048]),
#  ('base_model/conv2d_51/kernel', [3, 3, 2048, 2048]),
#  ('base_model/conv2d_52/kernel', [1, 1, 2048, 8192]),


#
# [('base_model/batch_normalization/beta', [256]),
#  ('base_model/batch_normalization/gamma', [256]),
#  ('base_model/batch_normalization/moving_mean', [256]),
#  ('base_model/batch_normalization/moving_variance', [256]),

#  ('base_model/batch_normalization_1/beta', [1024]),
#  ('base_model/batch_normalization_1/gamma', [1024]),
#  ('base_model/batch_normalization_1/moving_mean', [1024]),
#  ('base_model/batch_normalization_1/moving_variance', [1024]),

#  ('base_model/batch_normalization_2/beta', [256]),
#  ('base_model/batch_normalization_2/gamma', [256]),
#  ('base_model/batch_normalization_2/moving_mean', [256]),
#  ('base_model/batch_normalization_2/moving_variance', [256]),

#  ('base_model/batch_normalization_3/beta', [256]),
#  ('base_model/batch_normalization_3/gamma', [256]),
#  ('base_model/batch_normalization_3/moving_mean', [256]),
#  ('base_model/batch_normalization_3/moving_variance', [256]),

#  ('base_model/batch_normalization_4/beta', [1024]),
#  ('base_model/batch_normalization_4/gamma', [1024]),
#  ('base_model/batch_normalization_4/moving_mean', [1024]),
#  ('base_model/batch_normalization_4/moving_variance', [1024]),

#  ('base_model/batch_normalization_5/beta', [256]),
#  ('base_model/batch_normalization_5/gamma', [256]),
#  ('base_model/batch_normalization_5/moving_mean', [256]),
#  ('base_model/batch_normalization_5/moving_variance', [256]),

#  ('base_model/batch_normalization_6/beta', [256]),
#  ('base_model/batch_normalization_6/gamma', [256]),
#  ('base_model/batch_normalization_6/moving_mean', [256]),
#  ('base_model/batch_normalization_6/moving_variance', [256]),

#  ('base_model/batch_normalization_7/beta', [1024]),
#  ('base_model/batch_normalization_7/gamma', [1024]),
#  ('base_model/batch_normalization_7/moving_mean', [1024]),
#  ('base_model/batch_normalization_7/moving_variance', [1024]),

# ('base_model/batch_normalization_8/beta', [256]),
#  ('base_model/batch_normalization_8/gamma', [256]),
#  ('base_model/batch_normalization_8/moving_mean', [256]),
#  ('base_model/batch_normalization_8/moving_variance', [256]),

#  ('base_model/batch_normalization_9/beta', [256]),
#  ('base_model/batch_normalization_9/gamma', [256]),
#  ('base_model/batch_normalization_9/moving_mean', [256]),
#  ('base_model/batch_normalization_9/moving_variance', [256]),


#  ('base_model/batch_normalization_10/beta', [1024]),
#  ('base_model/batch_normalization_10/gamma', [1024]),
#  ('base_model/batch_normalization_10/moving_mean', [1024]),
#  ('base_model/batch_normalization_10/moving_variance', [1024]),

#  ('base_model/batch_normalization_11/beta', [2048]),
#  ('base_model/batch_normalization_11/gamma', [2048]),
#  ('base_model/batch_normalization_11/moving_mean', [2048]),
#  ('base_model/batch_normalization_11/moving_variance', [2048]),
#  ('base_model/batch_normalization_12/beta', [512]),
#  ('base_model/batch_normalization_12/gamma', [512]),
#  ('base_model/batch_normalization_12/moving_mean', [512]),
#  ('base_model/batch_normalization_12/moving_variance', [512]),
#  ('base_model/batch_normalization_13/beta', [512]),
#  ('base_model/batch_normalization_13/gamma', [512]),
#  ('base_model/batch_normalization_13/moving_mean', [512]),
#  ('base_model/batch_normalization_13/moving_variance', [512]),
#  ('base_model/batch_normalization_14/beta', [2048]),
#  ('base_model/batch_normalization_14/gamma', [2048]),
#  ('base_model/batch_normalization_14/moving_mean', [2048]),
#  ('base_model/batch_normalization_14/moving_variance', [2048]),
# ('base_model/batch_normalization_15/beta', [512]),
#  ('base_model/batch_normalization_15/gamma', [512]),
#  ('base_model/batch_normalization_15/moving_mean', [512]),
#  ('base_model/batch_normalization_15/moving_variance', [512]),
#  ('base_model/batch_normalization_16/beta', [512]),
#  ('base_model/batch_normalization_16/gamma', [512]),
#  ('base_model/batch_normalization_16/moving_mean', [512]),
#  ('base_model/batch_normalization_16/moving_variance', [512]),
#  ('base_model/batch_normalization_17/beta', [2048]),
#  ('base_model/batch_normalization_17/gamma', [2048]),
#  ('base_model/batch_normalization_17/moving_mean', [2048]),
#  ('base_model/batch_normalization_17/moving_variance', [2048]),
#  ('base_model/batch_normalization_18/beta', [512]),
#  ('base_model/batch_normalization_18/gamma', [512]),
#  ('base_model/batch_normalization_18/moving_mean', [512]),
#  ('base_model/batch_normalization_18/moving_variance', [512]),
#  ('base_model/batch_normalization_19/beta', [512]),
#  ('base_model/batch_normalization_19/gamma', [512]),
#  ('base_model/batch_normalization_19/moving_mean', [512]),
#  ('base_model/batch_normalization_19/moving_variance', [512]),

#  ('base_model/batch_normalization_20/beta', [2048]),
#  ('base_model/batch_normalization_20/gamma', [2048]),
#  ('base_model/batch_normalization_20/moving_mean', [2048]),
#  ('base_model/batch_normalization_20/moving_variance', [2048]),
#  ('base_model/batch_normalization_21/beta', [512]),
#  ('base_model/batch_normalization_21/gamma', [512]),
#  ('base_model/batch_normalization_21/moving_mean', [512]),
#  ('base_model/batch_normalization_21/moving_variance', [512]),
# ('base_model/batch_normalization_22/beta', [512]),
#  ('base_model/batch_normalization_22/gamma', [512]),
#  ('base_model/batch_normalization_22/moving_mean', [512]),
#  ('base_model/batch_normalization_22/moving_variance', [512]),
#  ('base_model/batch_normalization_23/beta', [2048]),
#  ('base_model/batch_normalization_23/gamma', [2048]),
#  ('base_model/batch_normalization_23/moving_mean', [2048]),
#  ('base_model/batch_normalization_23/moving_variance', [2048]),
#  ('base_model/batch_normalization_24/beta', [4096]),
#  ('base_model/batch_normalization_24/gamma', [4096]),
#  ('base_model/batch_normalization_24/moving_mean', [4096]),
#  ('base_model/batch_normalization_24/moving_variance', [4096]),
#  ('base_model/batch_normalization_25/beta', [1024]),
#  ('base_model/batch_normalization_25/gamma', [1024]),
#  ('base_model/batch_normalization_25/moving_mean', [1024]),
#  ('base_model/batch_normalization_25/moving_variance', [1024]),
#  ('base_model/batch_normalization_26/beta', [1024]),
#  ('base_model/batch_normalization_26/gamma', [1024]),
#  ('base_model/batch_normalization_26/moving_mean', [1024]),
#  ('base_model/batch_normalization_26/moving_variance', [1024]),
#  ('base_model/batch_normalization_27/beta', [4096]),
#  ('base_model/batch_normalization_27/gamma', [4096]),
#  ('base_model/batch_normalization_27/moving_mean', [4096]),
#  ('base_model/batch_normalization_27/moving_variance', [4096]),
#  ('base_model/batch_normalization_28/beta', [1024]),
#  ('base_model/batch_normalization_28/gamma', [1024]),
#  ('base_model/batch_normalization_28/moving_mean', [1024]),
#  ('base_model/batch_normalization_28/moving_variance', [1024]),
#  ('base_model/batch_normalization_29/beta', [1024]),
#  ('base_model/batch_normalization_29/gamma', [1024]),
#  ('base_model/batch_normalization_29/moving_mean', [1024]),
#  ('base_model/batch_normalization_29/moving_variance', [1024]),

#  ('base_model/batch_normalization_30/beta', [4096]),
#  ('base_model/batch_normalization_30/gamma', [4096]),
#  ('base_model/batch_normalization_30/moving_mean', [4096]),
#  ('base_model/batch_normalization_30/moving_variance', [4096]),
#  ('base_model/batch_normalization_31/beta', [1024]),
#  ('base_model/batch_normalization_31/gamma', [1024]),
#  ('base_model/batch_normalization_31/moving_mean', [1024]),
#  ('base_model/batch_normalization_31/moving_variance', [1024]),
#  ('base_model/batch_normalization_32/beta', [1024]),
#  ('base_model/batch_normalization_32/gamma', [1024]),
#  ('base_model/batch_normalization_32/moving_mean', [1024]),
#  ('base_model/batch_normalization_32/moving_variance', [1024]),
# ('base_model/batch_normalization_33/beta', [4096]),
#  ('base_model/batch_normalization_33/gamma', [4096]),
#  ('base_model/batch_normalization_33/moving_mean', [4096]),
#  ('base_model/batch_normalization_33/moving_variance', [4096]),
#  ('base_model/batch_normalization_34/beta', [1024]),
#  ('base_model/batch_normalization_34/gamma', [1024]),
#  ('base_model/batch_normalization_34/moving_mean', [1024]),
#  ('base_model/batch_normalization_34/moving_variance', [1024]),
#  ('base_model/batch_normalization_35/beta', [1024]),
#  ('base_model/batch_normalization_35/gamma', [1024]),
#  ('base_model/batch_normalization_35/moving_mean', [1024]),
#  ('base_model/batch_normalization_35/moving_variance', [1024]),
#  ('base_model/batch_normalization_36/beta', [4096]),
#  ('base_model/batch_normalization_36/gamma', [4096]),
#  ('base_model/batch_normalization_36/moving_mean', [4096]),
#  ('base_model/batch_normalization_36/moving_variance', [4096]),
#  ('base_model/batch_normalization_37/beta', [1024]),
#  ('base_model/batch_normalization_37/gamma', [1024]),
#  ('base_model/batch_normalization_37/moving_mean', [1024]),
#  ('base_model/batch_normalization_37/moving_variance', [1024]),
#  ('base_model/batch_normalization_38/beta', [1024]),
#  ('base_model/batch_normalization_38/gamma', [1024]),
#  ('base_model/batch_normalization_38/moving_mean', [1024]),
#  ('base_model/batch_normalization_38/moving_variance', [1024]),
#  ('base_model/batch_normalization_39/beta', [4096]),
#  ('base_model/batch_normalization_39/gamma', [4096]),
#  ('base_model/batch_normalization_39/moving_mean', [4096]),
#  ('base_model/batch_normalization_39/moving_variance', [4096]),

#  ('base_model/batch_normalization_40/beta', [1024]),
#  ('base_model/batch_normalization_40/gamma', [1024]),
#  ('base_model/batch_normalization_40/moving_mean', [1024]),
#  ('base_model/batch_normalization_40/moving_variance', [1024]),
#  ('base_model/batch_normalization_41/beta', [1024]),
#  ('base_model/batch_normalization_41/gamma', [1024]),
#  ('base_model/batch_normalization_41/moving_mean', [1024]),
#  ('base_model/batch_normalization_41/moving_variance', [1024]),
#  ('base_model/batch_normalization_42/beta', [4096]),
#  ('base_model/batch_normalization_42/gamma', [4096]),
#  ('base_model/batch_normalization_42/moving_mean', [4096]),
#  ('base_model/batch_normalization_42/moving_variance', [4096]),
#  ('base_model/batch_normalization_43/beta', [8192]),
#  ('base_model/batch_normalization_43/gamma', [8192]),
#  ('base_model/batch_normalization_43/moving_mean', [8192]),
#  ('base_model/batch_normalization_43/moving_variance', [8192]),
# ('base_model/batch_normalization_44/beta', [2048]),
#  ('base_model/batch_normalization_44/gamma', [2048]),
#  ('base_model/batch_normalization_44/moving_mean', [2048]),
#  ('base_model/batch_normalization_44/moving_variance', [2048]),
#  ('base_model/batch_normalization_45/beta', [2048]),
#  ('base_model/batch_normalization_45/gamma', [2048]),
#  ('base_model/batch_normalization_45/moving_mean', [2048]),
#  ('base_model/batch_normalization_45/moving_variance', [2048]),
#  ('base_model/batch_normalization_46/beta', [8192]),
#  ('base_model/batch_normalization_46/gamma', [8192]),
#  ('base_model/batch_normalization_46/moving_mean', [8192]),
#  ('base_model/batch_normalization_46/moving_variance', [8192]),
#  ('base_model/batch_normalization_47/beta', [2048]),
#  ('base_model/batch_normalization_47/gamma', [2048]),
#  ('base_model/batch_normalization_47/moving_mean', [2048]),
#  ('base_model/batch_normalization_47/moving_variance', [2048]),
#  ('base_model/batch_normalization_48/beta', [2048]),
#  ('base_model/batch_normalization_48/gamma', [2048]),
#  ('base_model/batch_normalization_48/moving_mean', [2048]),
#  ('base_model/batch_normalization_48/moving_variance', [2048]),
#  ('base_model/batch_normalization_49/beta', [8192]),
#  ('base_model/batch_normalization_49/gamma', [8192]),
#  ('base_model/batch_normalization_49/moving_mean', [8192]),
#  ('base_model/batch_normalization_49/moving_variance', [8192]),

#  ('base_model/batch_normalization_50/beta', [2048]),
#  ('base_model/batch_normalization_50/gamma', [2048]),
#  ('base_model/batch_normalization_50/moving_mean', [2048]),
#  ('base_model/batch_normalization_50/moving_variance', [2048]),
#  ('base_model/batch_normalization_51/beta', [2048]),
#  ('base_model/batch_normalization_51/gamma', [2048]),
#  ('base_model/batch_normalization_51/moving_mean', [2048]),
#  ('base_model/batch_normalization_51/moving_variance', [2048]),
#  ('base_model/batch_normalization_52/beta', [8192]),
#  ('base_model/batch_normalization_52/gamma', [8192]),
#  ('base_model/batch_normalization_52/moving_mean', [8192]),
#  ('base_model/batch_normalization_52/moving_variance', [8192]),


#  ('global_step', []),
#  ('head_supervised/linear_layer/dense/bias', [1000]),
#  ('head_supervised/linear_layer/dense/bias/Momentum', [1000]),
#  ('head_supervised/linear_layer/dense/kernel', [8192, 1000]),
#  ('head_supervised/linear_layer/dense/kernel/Momentum', [8192, 1000])]





