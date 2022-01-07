
import os
import argparse
from mindspore import context
import numpy as np
from easydict import EasyDict as edict
import mindspore.dataset as ds
from mindspore.train.callback import LossMonitor
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
import mindspore
# bp_cfg = edict({
#     'learning_rate': [0.1, 0.3, 0.5],
#     'momentum': 0.9,
#     'num_epochs': 100,
#     'input_size': 63,
#     'hidden_size': 6,
#     'output_size': 9,
#     'save_checkpoint_steps': 10000,
#     'keep_checkpoint_max': 1
# })

class DatasetGenerator:
    def __init__(self, path):
        self.data = np.random.sample((10, 63))
        self.label = np.random.sample((10, 1))
        
        # a = [
        #             [1.,0.,0.,0.,0.,0.,0.,0.,0.]
        #             [0.,1.,0.,0.,0.,0.,0.,0.,0.]
        #             [0.,0.,1.,0.,0.,0.,0.,0.,0.]
        #             [0.,0.,0.,1.,0.,0.,0.,0.,0.]
        #             [0.,0.,0.,0.,1.,0.,0.,0.,0.]
        #             [0.,0.,0.,0.,0.,1.,0.,0.,0.]
        #             [0.,0.,0.,0.,0.,0.,1.,0.,0.]
        #             [0.,0.,0.,0.,0.,0.,0.,1.,0.]
        #             [0.,0.,0.,0.,0.,0.,0.,0.,1.]
        #         ]
        for i in range(0, 10):
            name = '%d.txt' % i
            array = np.loadtxt(path + name, delimiter=',')
            
            self.data[i] = array
            self.label[i] = i
        self.data.dtype = 'float32'
        self.label.dtype = 'float32'
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

a = DatasetGenerator('/bp/train/')
# 数据集：
dataset = ds.GeneratorDataset(a, ["data", "label"], shuffle=False)
#print(a.__getitem__(1))

class BP(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(BP, self).__init__()
        # 定义所需要的运算
        self.input = nn.Dense(63, 10)
        self.hidden = nn.Dense(10, 20)
        self.output = nn.Dense(20, 1)

        self.relu = nn.ReLU()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = x.reshape(-1, 63)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

net = BP()
loss = nn.MSELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

from mindspore import Model
# 模型：
model = Model(net, loss, opt)
model.train(epoch=1, train_dataset=dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)

# 评估：
eval_net = nn.WithEvalCell(net, loss)
eval_net.set_train(False)

e = DatasetGenerator('/bp/test/')
eval_dataset = ds.GeneratorDataset(e, ["data", "label"], shuffle=False)

mae = nn.MAE()
loss = nn.Loss()

mae.clear()
loss.clear()
for d in eval_dataset.create_dict_iterator():
    outputs = eval_net(d["data"], d["label"])
    print(outputs)
    mae.update(outputs[1], outputs[2])
    loss.update(outputs[0])

mae_result = mae.eval()
loss_result = loss.eval()
print("mae: ", mae_result)
print("loss: ", loss_result)
print(net)
