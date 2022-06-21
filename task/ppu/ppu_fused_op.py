from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
class ForwardPPUFusedOp(Operator):
    def __init__(self,op_list):
        super().__init__(type=OperatorType.FORWARD,
                        name=unique_class_name(self))
        self.op_list = op_list
        self.bn = None
        self.relu = None
        self.add = None
        self.maxpool = None
        for op in self.op_list:
            class_name = type(op).__name__
            if class_name=="ForwardBatchnorm":
                self.bn = op
            elif class_name=="ForwardRelu":
                self.relu = op
            elif class_name=="ForwardAdd":
                self.add = op
            elif class_name=="ForwardMaxpool":
                self.maxpool = op
            else:
                assert False

    @classmethod
    def replace_from(self,find_ops):
        return ForwardPPUFusedOp(op_list = find_ops)

    def to_ppu_instr(self):
        
        bn_finish = 0
        if self.bn:
            bn_input_shape = self.bn.tensors.get("input").shape
            bn_finish = bn_input_shape[1]*bn_input_shape[2]*bn_input_shape[3] - 1
        res_acc_finish = 0
        if self.add:
            add_input_shape = self.add.tensors.get("input1").shape
            res_acc_finish = add_input_shape[1]*add_input_shape[2]*add_input_shape[3] - 1
        instruction = Instruction(config_path="task/ppu/ppu_control.yaml",name="ppu",init_data={
            #BatchNorm
            "bn_en":"enable" if self.bn else "bypass",
            "bn_direct":"forward",
            "bn_double_en":"single",
            # "bn_finish":bn_finish,
            #ReLU
            "act_fwd_en":"enable" if self.relu else "bypass",
            "act_bp_en": "bypass",
            #ResAcc
            "ResAcc_fwd_en": "enable" if self.add else "bypass",
            "ResAcc_bp_en": "bypass",
            "ResAcc_bp_single_line_en": "double",
            # "ResAcc_bp_finish": res_acc_finish,
            #Maxpool
            "pooling_fwd_en": "enable" if self.maxpool else "bypass",
            "pooling_fwd_mode": self.maxpool.attrs.get("kernel_size") if self.maxpool else 2,
            "pooling_bp_en": "bypass",
            "pooling_bp_mode": 2,
        })
        instruction.set("bn_finish", int_to_bits(bn_finish,16).to01(),use_bits=True)
        instruction.set("ResAcc_bp_finish", int_to_bits(res_acc_finish,16).to01(),use_bits=True)
        # bits = instruction.export()

        return instruction
    

class BackwardPPUFusedOp(Operator):
    def __init__(self,op_list):
        super().__init__(type=OperatorType.FORWARD,
                        name=unique_class_name(self))
        self.op_list = op_list
        self.bn = None
        self.relu = None
        self.add = None
        self.maxpool = None
        for op in self.op_list:
            class_name = type(op).__name__
            if class_name=="BackwardBatchnorm":
                self.bn = op
            elif class_name=="BackwardRelu":
                self.relu = op
            elif class_name=="BackwardSplit" or class_name=="BackwardScalarAdd":
                self.add = op
                self.scalar_add = class_name=="BackwardScalarAdd"
            elif class_name=="BackwardMaxpool":
                self.maxpool = op
            else:
                assert False,class_name

    @classmethod
    def replace_from(self,find_ops):
        return BackwardPPUFusedOp(op_list = find_ops)

    def to_ppu_instr(self):
        bn_finish = 0
        if self.bn:
            bn_input_shape = self.bn.tensors.get("input_grad").shape
            bn_finish = bn_input_shape[1]*bn_input_shape[2]*bn_input_shape[3]-1

        bn_double_en = "single"
        bns = [op for op in self.op_list if type(op).__name__=="BackwardBatchnorm"]
        if len(bns)==2:
            bn_double_en = "double"
        print(bn_double_en)
        res_acc_finish = 0
        if self.add:
            add_input_shape = self.add.tensors.get("input_grad").shape
            res_acc_finish = add_input_shape[1]*add_input_shape[2]*add_input_shape[3]-1
        instruction = Instruction(config_path="task/ppu/ppu_control.yaml",name="ppu",init_data={
            #BatchNorm
            "bn_en":"enable" if self.bn else "bypass",
            "bn_direct":"backward",
            "bn_double_en":bn_double_en,
            #ReLU
            "act_fwd_en": "bypass",
            "act_bp_en": "enable" if self.relu else "bypass",
            #ResAcc
            "ResAcc_fwd_en": "bypass",
            "ResAcc_bp_en": "enable" if self.add else "bypass",
            "ResAcc_bp_single_line_en": "single" if self.add and self.scalar_add else "double",
            #Maxpool
            "pooling_fwd_en": "bypass",
            "pooling_fwd_mode": self.maxpool.attrs.get("kernel_size") if self.maxpool else 2,
            "pooling_bp_en": "enable" if self.maxpool else "bypass",
            "pooling_bp_mode": self.maxpool.attrs.get("kernel_size") if self.maxpool else 2,
        })
        instruction.set("bn_finish", int_to_bits(bn_finish,16).to01(),use_bits=True)
        instruction.set("ResAcc_bp_finish", int_to_bits(res_acc_finish,16).to01(),use_bits=True)
        # bits = instruction.export()

        return instruction

class CrossEntropyLoss(Operator):
    def __init__(self,forward_softmax,
                    forward_entropy,
                    backward_softmax,
                    backward_entropy):
        super().__init__(type=OperatorType.FORWARD,
                        name=unique_class_name(self))
        self.forward_softmax = forward_softmax
        self.forward_entropy = forward_entropy
        self.backward_softmax = backward_softmax
        self.backward_entropy = backward_entropy

    @classmethod
    def replace_from(self,find_ops):
        forward_softmax,forward_entropy,backward_entropy,backward_softmax = find_ops
        return CrossEntropyLoss(forward_softmax=forward_softmax,
                                forward_entropy=forward_entropy,
                                backward_softmax=backward_softmax,
                                backward_entropy=backward_entropy)