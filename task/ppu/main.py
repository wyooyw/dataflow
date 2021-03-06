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

def run(verbose, module, use_gpu, use_half, load_params):
    """ Generate test data of ppu.

    Params:
        verbose(boolean): Show detail.
        module(str): which
    """

    # Check gpu and fp16 avaliable
    if use_gpu==False and use_half==True:
        assert False,"You must use gpu when you want to use fp16. Add '-g' in your command."
    if use_gpu==True:
        assert torch.cuda.is_available(),"Gpu is not available."
    print(f"Use gpu:{use_gpu}\tUse half:{use_half}")


    # Init nureal network
    torch_net = {
        "resnet":resnet18_cifar,
        "alexnet":AlexNet
    }[module]()
    if not load_params==None:
        device = torch.device('cuda') if use_gpu else torch.device('cpu')
        sd = torch.load(load_params,map_location=device)
        
        state_dict = OrderedDict()
        for key,value in sd["state_dict"].items():
            state_dict[key.replace("module.","")] = value
        torch_net.load_state_dict(state_dict)
    if use_gpu:
        torch_net = torch_net.cuda()
        if use_half:
            torch_net = torch_net.half()
    torch_net.eval()

    #Merge weight,bias into mean,var
    # remove_bn_affine(torch_net)
    
    in_shape=[4,3,32,32]
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
    print(net)
    # show_memory(net)
    
    # import sys
    # sys.exit()
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
    def load_data(n):
        images = []
        labels = []
        for i in range(0,n):
            image,label = train_dataset[i]
            image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])#.half()
            zero = torch.tensor([[0.0]*10]).reshape(1,10)
            zero[0] = 1.0
            images.append(image)
            labels.append(zero)
        images = torch.cat(images,0)
        labels = torch.cat(labels,0)
        return images,labels

    input,label = load_data(4)

    if use_gpu:
        input = input.cuda()
        label = label.cuda()
    if use_half:
        input = input.half()
        label = label.half()


    input.requires_grad=True
    executer = Executer(net)
    output = executer.execute(input,label,to="WGConv_0").tensors.get_data("weight_grad")
    # output = executer.execute(input,label,to="FLinear_2").tensors.get_data("output")

    torch_output = torch_net(input)
    torch_output = torch.nn.CrossEntropyLoss()(torch_output,label)
    torch_output.backward()
    # torch_output = input.grad
    torch_output = torch_net.conv1.weight.grad
    # print(net)
    # print(output)
    # print(output[0,:,:,:])
    # print(torch_output[0,:,:,:])
    if output.shape==torch_output.shape:
        # print(output)
        # print(label)
        print(torch.max(output),torch.max(torch_output))
        print(torch.min(output),torch.min(torch_output))
        print(torch.mean(torch.abs(output-torch_output)))
        print(torch.max(torch.abs(output-torch_output)))
        # print(torch.mean(torch.abs(output-torch_output))<0.01)
    else:
        print(f"Shape is not equal! output.shape={output.shape}, torch_output.shape={torch_output.shape}")
    
    # net.statistic_op()
# if __name__=="__main__":
#     run()