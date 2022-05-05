import torch
import torch.nn as nn
from torch.nn import Module
import torch.fx as fx
from torch.fx import symbolic_trace,Graph,Node,replace_pattern

from compiler.generate.op.conv import DualConv
from compiler.generate.op.softmax import DualSoftmax
from compiler.generate.op.entropy import DualEntropy
from compiler.generate.op.relu import DualRelu
from compiler.generate.op.flatten import DualFlatten
from compiler.generate.op.linear import DualLinear
from compiler.generate.net.net import Net

class Converter(object):
    """将pytorch模型转换为sparsetrain-IR
    """
    def __init__(self,module,in_shape=[32,3,32,32]):
        self.origin_module = module
        self.trace = symbolic_trace(self.origin_module)
        self.named_modules = dict(self.trace.named_modules())
        self.in_shape = in_shape
        self.net = Net()
        self.map = dict() #key:original ;value :(forward_new,backward_new)
        pass

    def get_in_shape(self,node):
        in_shape = []
        assert len(node.args)>0,"Length of node.arg should bigger than 0."
        arg = node.args[0]
        # if str(node)=="relu":
        #     import ipdb
        #     ipdb.set_trace()
        if arg.op=="placeholder":
            in_shape = self.in_shape
        if arg.op=="call_module" and arg.target in self.named_modules:
            dual = self.map[self.named_modules[arg.target]]
            in_shape = dual.forward.out_shape
        assert len(in_shape)>0,f"Get in_shape error.node:{node}"
        return in_shape
    
    def convert(self):
        graph = self.trace.graph
        named_modules = self.named_modules
        for node in graph.nodes:
            
            if node.op == 'call_module':
                module = named_modules[node.target]
                in_shape = self.get_in_shape(node)

                if type(module) == nn.Conv2d:
                    conv_dual = DualConv.from_torch_module(in_shape=in_shape,module=module)
                    self.net.add_operator(conv_dual.forward,conv_dual.backward)
                    self.map[module] = conv_dual
                    #加入前驱节点
                    for arg in node.args:
                        if arg.target in named_modules:
                            dual = self.map[named_modules[arg.target]]
                            dual.forward.connect_successor(conv_dual.forward)
                            dual.backward.connect_predecessor(conv_dual.backward)
                elif type(module) == nn.ReLU:
                    relu_dual = DualRelu.from_torch_module(in_shape=in_shape,module=module)
                    self.net.add_operator(relu_dual.forward,relu_dual.backward)
                    self.map[module] = relu_dual
                    #加入前驱节点
                    for arg in node.args:
                        if arg.target in named_modules:
                            dual = self.map[named_modules[arg.target]]
                            dual.forward.connect_successor(relu_dual.forward)
                            dual.backward.connect_predecessor(relu_dual.backward)
                elif type(module) == nn.Flatten:
                    flatten_dual = DualFlatten.from_torch_module(in_shape=in_shape,module=module)
                    self.net.add_operator(flatten_dual.forward,flatten_dual.backward)
                    self.map[module] = flatten_dual
                    #加入前驱节点
                    for arg in node.args:
                        if arg.target in named_modules:
                            dual = self.map[named_modules[arg.target]]
                            dual.forward.connect_successor(flatten_dual.forward)
                            dual.backward.connect_predecessor(flatten_dual.backward)
                elif type(module) == nn.Linear:
                    flatten_dual = DualLinear.from_torch_module(in_shape=in_shape,module=module)
                    self.net.add_operator(flatten_dual.forward,flatten_dual.backward)
                    self.map[module] = flatten_dual
                    #加入前驱节点
                    for arg in node.args:
                        if arg.target in named_modules:
                            dual = self.map[named_modules[arg.target]]
                            dual.forward.connect_successor(flatten_dual.forward)
                            dual.backward.connect_predecessor(flatten_dual.backward)
            elif node.op=="output":
                assert len(node.args)==1,"Convert output error."
                in_shape = self.get_in_shape(node)
                dual_last = self.map[named_modules[node.args[0].target]]
                dual_softmax = DualSoftmax.from_torch_module(in_shape=in_shape,module=None)
                dual_entropy = DualEntropy.from_torch_module(in_shape=in_shape,module=None)
                self.net.add_operator(dual_softmax.forward,dual_softmax.backward)
                self.net.add_operator(dual_entropy.forward,dual_entropy.backward)
                
                dual_last.forward.connect_successor(dual_softmax.forward)
                dual_softmax.forward.connect_successor(dual_entropy.forward)
                dual_last.backward.connect_predecessor(dual_softmax.backward)
                dual_softmax.backward.connect_predecessor(dual_entropy.backward)

                dual_entropy.forward.connect_successor(dual_entropy.backward)
        
        

class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4,5,5)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__=="__main__":
    net = MyNet()
    converter = Converter(net,in_shape=[32,3,32,32])
    converter.convert()
    print(converter.net)

