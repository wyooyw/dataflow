from compiler.target_gen.memory.storage import Storage
from compiler.utils import unique_class_name
import copy
class Tensor:
    # count = 0
    def __init__(self,storage:Storage,
                        shape:tuple[int],
                        addr,
                        offset,
                        stride:tuple[int]):
        self.addr = addr
        self.storage = storage

        self.shape = shape
        self.ndim = len(shape)
        self.offset = offset
        self.stride = stride
        self.name = unique_class_name(self)
        # self.id = Tensor.count
        # Tensor.count += 1

    def __copy__(self):
        return type(self)(storage=self.storage,
                            shape=self.shape.copy(),
                            addr=self.addr,
                            offset=self.offset,
                            stride=self.stride.copy())

    def __str__(self):
        return f"[tensor] name={self.name} shape={self.shape} offset={self.offset}"
    
    def export(self):
        return [self.ndim,self.shape,self.storage.addr]

    def __getitem__(self,key):
        
        def convert_key(key):
            rst = [(0,shape) for shape in self.shape]
            if not type(key)==tuple:
                key = [key]
            for idx,item in enumerate(key):
                if type(item)==int:
                    rst[idx] = (item,item+1)
                else:
                    rst[idx] = (item.start,item.stop)
                assert rst[idx][0]>=0 and rst[idx][1]<=self.shape[idx],f"[{self.name}] Index of dim {idx} should in [{0},{self.shape[idx]}], but got [{rst[idx][0],rst[idx][1]}]"
            return rst
        key = convert_key(key)
        new_tensor = copy.copy(self)
        assert len(key)==len(new_tensor.stride)

        #修改offet
        offset_delta = 0
        for idx,item in enumerate(key):
            offset_delta += item[0] * new_tensor.stride[idx]
        new_tensor.offset += offset_delta

        #修改shape
        for idx,item in enumerate(key):
            left,right = item
            new_tensor.shape[idx] = right - left
        
        return new_tensor
    
            

        