from compiler.graph_ir import Dual,Operator,OperatorType
from compiler.graph_ir.attrs.add_attrs import *
from compiler.graph_ir.tensors.add_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn
import torch
from simulator.memory import Memory

class ScalarAddTensors(Tensors):
    def __init__(self,output_grad,output_grad_res,bn_std,input_grad):
        super().__init__()
        self.tensors["output_grad"] = output_grad
        self.tensors["output_grad_res"] = output_grad_res
        self.tensors["bn_std"] = bn_std
        self.tensors["input_grad"] = input_grad
        self.add_read_tensor("output_grad")
        self.add_read_tensor("output_grad_res")
        self.add_write_tensor("bn_std")
        self.add_write_tensor("input_grad")
        self.input = [output_grad,output_grad_res]
        self.output = input_grad

class BackwardScalarAdd(Operator):
    def __init__(self,add):
        super().__init__(type=OperatorType.BACKWARD,
                        name=unique_class_name(self))
                        
        self.tensors = ScalarAddTensors(output_grad=add._main_add_tensor,
                                    output_grad_res=add._bn_input_grad,
                                    bn_std=add._bn_std,
                                    input_grad=add.tensors.get("input_grad"))
        
    @classmethod
    def replace_from(self,add):
        return BackwardScalarAdd(add = add)