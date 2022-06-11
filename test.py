import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.target_gen.memory.segment import Segment,SegmentType
from compiler.target_gen.memory.tensor import Tensor
from compiler.target_gen.memory.storage import Storage
from compiler.generate.net.net import Net
from compiler.generate.op.conv import DualConv,ForwardConv
from compiler.generate.op.relu import DualRelu,ForwardRelu
from compiler.generate.op.add import ForwardAdd
from compiler.generate.op.batchnorm import ForwardBatchnorm,BackwardBatchnorm
from compiler.scheduler.normal_scheduler import NormalScheduler
from compiler.scheduler.wu_imm_scheduler import WUImmScheduler
from simulator.memory import Memory
# from compiler.generate.op.split import SplitDualGenerator
import torch.nn as nn
from model.resnet import resnet18_cifar
from model.lenet import *
from model.alexnet import AlexNet

# from compiler.generate.op.aggregate import AggregateDualGenerator
from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.replace_tool import Finder,ReplaceTool
from compiler.config import Config,CodeGen
from converter import Converter

from compiler.target_gen.memory.storage import StorageType

from backends.sparse_train.target_code.instruction_gen import InstructionGenerator
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
        self.conv1 = nn.Conv2d(3,4,5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3,4,5)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.bn2 = nn.BatchNorm2d(num_features=4)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,10)
    def forward(self,x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.maxpool1(y)
        y = self.bn1(y)
        z = self.conv2(x)
        z = self.relu2(z)
        z = self.maxpool2(z)
        z = self.bn2(z)
        o = y + z
        o = self.flatten(o)
        # y = self.fc1(y)
        o = self.fc2(o)
        return o
        
def testConverter():

    net = resnet18_cifar()
    converter = Converter(net,in_shape=[32,3,32,32])
    converter.convert()
    print(converter.net)
    # named_modules = dict(converter.trace.named_modules())
    # node_op = set()
    # module_target = set()
    # function_target = set()
    # conv_num = 0
    # for node in converter.trace.graph.nodes:
    #     node_op.add(node.op)
    #     if node.op=="call_module":
    #         target_type = str(type(named_modules.get(node.target,"missing")))
    #         module_target.add(target_type)
    #         if target_type=="<class 'torch.nn.modules.conv.Conv2d'>":
    #             conv_num+=1
    #     elif node.op=="call_function":
    #         function_target.add(str(node.target))
    # print("node_op",node_op)
    # print("module_target",module_target)
    # print("function_target",function_target)
    # print("conv_num",conv_num)

def testMerger():
    net = resnet18_cifar()
    converter = Converter(net,in_shape=[4,3,32,32])
    converter.convert()
    net = converter.net
    replace_tool = ReplaceTool(net=net,config_path="./backends/sparse_train/replace.yaml")
    replace_tool.replace_all()
    print(net)
    # print("tensor num:",MemoryManager().get_tensor_num())
    # size = MemoryManager().get_tensor_size()
    # print("tensor size:",size," item")
    # print("tensor size(all fp16):",size*2/1024/1024," MB")
    # print("tensor size(all fp32):",size*4/1024/1024," MB")
    # print("tensor size max:",MemoryManager().get_tensor_max()," items")

def testMemoryManage():
    torch_net = TestNet4()
    # torch_net = resnet18_cifar()
    # torch_net = AlexNet()
    # net = torchvision.

    # x = torch.ones(1,3,32,32)
    # out = net(x)
    
    total_params = sum(p.numel() for p in torch_net.parameters())
    in_shape=[3,3,4,4]
    print("in_shape:",in_shape)
    converter = Converter(torch_net,in_shape=in_shape)
    # print(converter.trace.graph)
    converter.convert()
    net = converter.net
    
    # print(net)
    
    #将BN算子的avg,std,alpha,beta合并成一个tensor
    #编译器里不好实现，这里hack一下
    # tmp = {}
    # for op in net.topo():
    #     if type(op)==ForwardBatchnorm or type(op)==BackwardBatchnorm:
    #         avg_storage = op.tensors.get("avg").storage
    #         if avg_storage in tmp:
    #             bn_use = tmp[avg_storage]
    #         else:
    #             bn_use = MemoryManager().allocWeight((4,op.tensors.get("avg").shape[0]))
    #             tmp[avg_storage] = bn_use
    #         op.tensors.set("bn_use",bn_use)
    #         op.tensors.add_read_tensor("bn_use")
    #         op.tensors.tensors.pop("avg")
    #         op.tensors.tensors.pop("std")
    #         op.tensors.tensors.pop("alpha")
    #         op.tensors.tensors.pop("beta")

    replace_tool = ReplaceTool(net=net,config_path="./backends/sparse_train/replace.yaml")
    # replace_tool.replace_all()
    # print("======================== Net =========================")
    # print(net)
    # print("\n======================== Memory =========================")
    scheduler = NormalScheduler()
    # scheduler = WUImmScheduler()
    scheduler.schedule(net)
    print(net)
    
    storage_record = {
        StorageType.ACTIVATION:[],
        StorageType.FEATURE_GRAD:[],
        StorageType.WEIGHT_GRAD:[],
        StorageType.WEIGHT:[]
    }
    storage_stats = {
        StorageType.ACTIVATION:0,
        StorageType.FEATURE_GRAD:0,
        StorageType.WEIGHT_GRAD:0,
        StorageType.WEIGHT:0
    }
    storage_visit = {}
    for op in net.topo():
        print(f"{op.name}:")
        for key,tensor in op.get_tensors().tensors.items():
            if tensor:
                storage = tensor.storage
                if storage not in storage_visit:
                    storage_visit[storage] = f"{op.name}.{key}"
                    print(f"  [{storage.type}] {key} shape={tensor.shape} size={storage.size}")

                    storage_record[storage.type].append(storage)
                    storage_stats[storage.type] += storage.size
                else:
                    # pass
                    print(f"  [{storage.type}] {key} (share storage with {storage_visit[storage]})")
            else:
                # pass
                print(f"  [None] {key}")
    total = 0
    for key,stats in storage_stats.items():
        total += stats
        b = stats*2
        kb = b/1024
        mb = kb/1024
        b = int(b*100)/100
        kb = int(kb*100)/100
        mb = int(mb*100)/100
        print(f"{key}:{stats} num, {b}B = {kb}KB = {mb}MB")
    # print("total num:",total)
    print(storage_stats)
    
    # print(net.count())
    net.reduce_tensor()
    print(net.count())
    net.set_tensor_index()

    # for op,tensor_name,tensor in net.all_tensors():
    #     print(op.name,tensor_name,tensor.index)

    # instr_gen = InstructionGenerator(net)
    # for instr in instr_gen.instruction_list:
    #     print(instr)
    # print(f'Pytorch say: {total_params} total parameters.')
    # MemoryManager().tensor_memory_layout2(net,show_image=True,save_image=True)
    MemoryManager().tensor_memory_layout2(net)
    from functools import reduce
    # input = torch.range(1.0,reduce(lambda x,y:x*y,in_shape)).reshape(in_shape)
    input = torch.randn(in_shape)
    input.requires_grad=True

    output = Memory().get(net.sim_run_to(input,"BBatchnorm_0").tensors.get("input_grad").addr)
    torch_output = torch_net(input)
    torch_output = torch.sum(torch_output)
    torch_output.backward()
    torch_output = input.grad
    print("my   :",output)
    print("torch:",torch_output)
    if output.shape==torch_output.shape:
        print(torch.max(torch.abs(output-torch_output))<0.01)
    else:
        print(f"Shape is not equal! output.shape={output.shape}, torch_output.shape={torch_output.shape}")
        
    # MemoryManager().count_read_and_write_times(net)
    # print(net.hash["BConv_1"].get_tensors().tensors["weight"].storage.data)
    # print(net.hash["BConv_0"].get_tensors().tensors["weight"].storage.data)
    
if __name__=="__main__":
    # testPointer()
    # testDualGenerator()
    # testMemoryManager()
    # testConfig()
    # testTensor()
    # testMerger()
    testMemoryManage()
    # net = Net()
    # converter = Converter(net,in_shape=[4,3,32,32])
    # print(converter.trace.graph)
    # print(Config().load("backends/sparse_train/replace.yaml"))