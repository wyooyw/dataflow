from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.target_gen.memory.segment import Segment,SegmentType
from compiler.target_gen.memory.tensor import Tensor
from compiler.target_gen.memory.storage import Storage
from compiler.generate.net.net import Net
from compiler.generate.op.conv import DualConv,ForwardConv
from compiler.generate.op.relu import DualRelu,ForwardRelu
from compiler.generate.op.add import ForwardAdd
# from compiler.generate.op.split import SplitDualGenerator
from backends.sparse_train.op.conv_relu import ForwardConvRelu
import torch.nn as nn

# from compiler.generate.op.aggregate import AggregateDualGenerator
from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.replace_tool import Finder,ReplaceTool
from compiler.config import Config,CodeGen
from converter import Converter
def testMemoryManager():
    mem = MemoryManager()
    mem.calcBases()
    for name,segment in mem.segments.items():
        print(f"name:{name}")
        print(f"\tsegment.base:{segment.base},segment.size:{segment.size}\n")

# def testDualGenerator():
#     conv_forward,conv_backward = ConvDualGenerator(in_batch=1,
#                                                     in_width=4,
#                                                     in_height=4,
#                                                     in_channels=3,
#                                                     out_channels=2,
#                                                     kernel_size=2,
#                                                     padding=0).getDual()
#     relu_forward,relu_backward = ReluDualGenerator(in_batch=1,
#                                                     in_width=3,
#                                                     in_height=3,
#                                                     in_channels=2).getDual()
#     conv_forward.next_output_to(relu_forward)
#     conv_forward.connect_successor(relu_forward)

#     conv2_forward,conv2_backward = ConvDualGenerator(in_batch=1,
#                                                     in_width=3,
#                                                     in_height=3,
#                                                     in_channels=2,
#                                                     out_channels=2,
#                                                     kernel_size=3,
#                                                     padding=1).getDual()
    
#     relu_forward.next_output_to(conv2_forward)
#     relu_forward.connect_successor(conv2_forward)

#     relu2_forward,relu2_backward = ReluDualGenerator(in_batch=1,
#                                                     in_width=3,
#                                                     in_height=3,
#                                                     in_channels=2).getDual()
#     conv2_forward.next_output_to(relu2_forward)
#     conv2_forward.connect_successor(relu2_forward)

#     net = Net()
#     for op in [conv_forward,relu_forward,conv2_forward,relu2_forward]:
#         net.add_operator(op)

#     print(net)

#     pattern = [ForwardConv2d,ForwardRelu]
#     finder = Finder(net,pattern)

#     find = finder.find()
#     for f in find:
#         fuse_conv_relu = ForwardFuseConvRelu.merge(conv=f[0],relu=f[1])
#     print("---------------")
#     print(net)

# def testPointer():
#     from compiler.utils.pointer import Pointer
#     ptr = Pointer(None)
#     ptr2 = ptr
#     print(ptr,ptr2)
#     ptr.set("helloworld")
#     print(ptr,ptr2)

class MyClass:
    def __init__(self):
        # print(unique_class_name(self))
        pass

def testConfig():
    codegen = CodeGen()
    codegen.generate_operator()
def testTensor():
    storage = Storage(100,[],0)
    tensor = Tensor(storage=storage,shape=[2,3,4,4],addr=-1,offset=0,stride=[48,16,4,1])
    new_tensor = tensor[0:2,1:3,2:4,1:4]
    new_new_tensor = new_tensor[0:1,0:2,0:1,1:3]
    print(tensor,tensor.storage)
    print(new_tensor,new_tensor.storage)
    print(new_new_tensor,new_new_tensor.storage)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(3,4,5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4,5,5)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,10)
    def forward(self,x):
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        y = self.conv2(x)
        y = self.relu2(y)
        y = y + x
        y = self.flatten(y)
        # y = self.fc1(y)
        y = self.fc2(y)
        return y
        
def testConverter():

    net = MyNet()
    converter = Converter(net,in_shape=[32,3,32,32])
    converter.convert()
    print(converter.net)

def testMerger():
    net = MyNet()
    converter = Converter(net,in_shape=[32,3,32,32])
    converter.convert()
    net = converter.net

    replace_tool = ReplaceTool(net=net,config_path="./backends/sparse_train/replace.yaml")
    replace_tool.replace_all()
    print(net)
if __name__=="__main__":
    # testPointer()
    # testDualGenerator()
    # testMemoryManager()
    # testConfig()
    # testTensor()
    testMerger()