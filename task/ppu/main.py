import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.scheduler.normal_scheduler import NormalScheduler
from compiler.scheduler.wu_imm_scheduler import WUImmScheduler
from simulator.memory import Memory
import torch.nn as nn
from model.resnet import resnet18_cifar
from model.lenet import *
from model.alexnet import AlexNet

from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.replace_tool import Finder,ReplaceTool
from compiler.config import Config,CodeGen
from converter import Converter

from backends.sparse_train.target_code.instruction_gen import InstructionGenerator

from compiler.target.dataflow import Dataflow
import numpy as np
from executer.executer import Executer
import task.ppu.execute_functions
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from task.ppu.utils import judge_single_line,replace,remove_bn_affine
from compiler.target_gen.memory.storage import StorageType

def show_memory(net):
    print("\n======================== Memory =========================")
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

def run():
    use_half = False
    use_gpu = False
    assert ((not use_gpu) and (not use_half)) or (use_gpu and torch.cuda.is_available())


    # Load state_dict
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sd = torch.load('model/resnet18_baseline_new_archi3_bnAffineTrue_fp16/model_best.pth.tar',map_location=device)
    
    state_dict = OrderedDict()
    for key,value in sd["state_dict"].items():
        state_dict[key.replace("module.","")] = value

    # Init nureal network
    torch_net = resnet18_cifar()
    torch_net.load_state_dict(state_dict)
    if use_gpu:
        torch_net = torch_net.cuda()
    if use_half:
        torch_net = torch_net.half()

    torch_net.eval()

    #Merge weight,bias into mean,var
    # remove_bn_affine(torch_net)
    
    in_shape=[1,3,32,32]
    print("in_shape:",in_shape)
    converter = Converter(torch_net,in_shape=in_shape)
    converter.convert()
    
    net = converter.net
    net.reduce_tensor()
    net.deal_input_data()
    replace(net)
    
    replace_tool = ReplaceTool(net=net,config_path="./task/ppu/merge.yaml")
    replace_tool.replace_all()
    scheduler = NormalScheduler()
    # scheduler = WUImmScheduler()
    scheduler.schedule(net)
    # print(net)
    # show_memory(net)
    
    
    net.set_tensor_index()

    # MemoryManager().tensor_memory_layout2(net)
    train_transformer = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean = (0.4914, 0.4822, 0.4465),
                                  std = (0.2023, 0.1994, 0.2010)),
            ])
    train_dataset = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=train_transformer)
    image,label = train_dataset[1]
    image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])#.half()
    zeros = torch.tensor([0.0,0,0,0,0,0,0,0,0,0])
    zeros[0] = 1.0
    label = zeros.reshape(1,-1)
    
    input = image#(torch.range(1.0,3*32*32)/512).reshape(1,3,32,32)
    label = label#torch.tensor([[1.0,0,0,0,0,0,0,0,0,0]])

    if use_gpu:
        input = input.cuda()
        label = label.cuda()
    if use_half:
        input = input.half()
        label = label.half()


    input.requires_grad=True
    executer = Executer(net)
    output = executer.execute(input,label,to="BEdge_0").tensors.get_data("output_grad")
    # output = executer.execute(input,label,to="FLinear_0").tensors.get_data("output")

    torch_output = torch_net(input)
    torch_output = torch.nn.CrossEntropyLoss()(torch_output,label)
    torch_output.backward()
    torch_output = input.grad
    # torch_output = torch_net.conv1.weight.grad
    # print(net)
    # print(output)
    # print(output)
    # print(torch_output)
    if output.shape==torch_output.shape:
        # print(output)
        # print(label)
        print(torch.max(torch.abs(output)))
        print(torch.max(torch.abs(torch_output)))
        print(torch.max(torch.abs(output-torch_output)))
        print(torch.max(torch.abs(output-torch_output))<0.01)
    else:
        print(f"Shape is not equal! output.shape={output.shape}, torch_output.shape={torch_output.shape}")
    
    # net.statistic_op()
# if __name__=="__main__":
#     run()