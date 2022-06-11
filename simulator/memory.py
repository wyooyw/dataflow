import torch
from compiler.utils import singleton
@singleton
class Memory(object):
    def __init__(self):
        self.memory = dict()
    
    def set(self,begin_addr,tensor,name="unknown"):
        size = tensor.numel() * 2
        end_addr = begin_addr + size - 1
        rm_addr = []
        for _addr,_tensor in self.memory.items():
            _begin_addr = _addr
            _end_addr = _begin_addr + _tensor.numel() * 2 - 1
            if not (_begin_addr > end_addr or  begin_addr > _end_addr):
                rm_addr.append(_addr)
            
        for addr in rm_addr:
            _tensor = self.memory[addr]
            print(f"\npop at {addr}B to {addr+_tensor.numel()*2-1}B , size = {_tensor.numel()*2}B, shape = {_tensor.shape}")
            # print(_tensor)
            self.memory.pop(addr)

        self.memory[begin_addr] = tensor
        print(f"\nadd at {begin_addr}B to {begin_addr+tensor.numel()*2-1}B , size = {tensor.numel()*2}B, shape = {tensor.shape}")
        # print(tensor)
    
    def get(self,begin_addr):
        if begin_addr in self.memory:
            return self.memory[begin_addr]
        else:
            assert False,f"{begin_addr} is not find!"

if __name__=="__main__":
    tensor1 = torch.randn(1,1)
    tensor1._name = "tensor1"
    tensor2 = torch.randn(2,4)
    tensor2._name = "tensor2"
    memory = Memory()
    memory.add(0,tensor1)
    memory.add(18,tensor2)