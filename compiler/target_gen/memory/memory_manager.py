from compiler.utils.singleton import singleton
from compiler.target_gen.memory.net import Net
from compiler.target_gen.memory.mem_operator import MemOperator
from compiler.target_gen.memory.storage import Storage,StorageType
from compiler.target_gen.memory.tensor import Tensor
from compiler.target_gen.memory.segment import Segment,SegmentType

from functools import reduce

@singleton
class MemoryManager(object):
    def __init__(self):
        # self.net = Net()
        self.tensor_list = []
        self.segments = {
            SegmentType.WEIGHT_STORAGE:Segment(type=SegmentType.WEIGHT_STORAGE),
            SegmentType.ACTIVATION_STORAGE:Segment(type=SegmentType.ACTIVATION_STORAGE),
            SegmentType.GRAD_STORAGE:Segment(type=SegmentType.GRAD_STORAGE),
            SegmentType.TENSOR:Segment(type=SegmentType.TENSOR),
            SegmentType.OPERATOR:Segment(type=SegmentType.OPERATOR),
            SegmentType.NET:Segment(type=SegmentType.NET)
        }
        pass
    
    def allocTensor(self,
                    shape,
                    storage_segment:Segment,
                    content=None):
        size = reduce(lambda x, y: x*y, shape)
        storage = Storage(size=size,content=content)
        tensor = Tensor(storage=storage,shape=shape)
        self.tensor_list.append(tensor)
        storage_segment.size += size

        tensor_segment = self.segments[SegmentType.TENSOR]
        ndim = len(shape)
        tensor_segment.size += 1 + ndim + 1 #1:ndim; ndim:shape; 1:storage_addr
        return tensor

    def allocWeight(self,shape,content=None):
        segment = self.segments[SegmentType.WEIGHT_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,content=content)
    
    def allocActivation(self,shape,content=None):
        segment = self.segments[SegmentType.ACTIVATION_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,content=content)
    
    def allocGrad(self,shape,content=None):
        segment = self.segments[SegmentType.GRAD_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,content=content)
    
    def calcBases(self):
        self.segments[SegmentType.WEIGHT_STORAGE].base = 0
        self.segments[SegmentType.ACTIVATION_STORAGE].base = self.segments[SegmentType.WEIGHT_STORAGE].base + \
                                                                self.segments[SegmentType.WEIGHT_STORAGE].size
        self.segments[SegmentType.GRAD_STORAGE].base = self.segments[SegmentType.ACTIVATION_STORAGE].base + \
                                                                self.segments[SegmentType.ACTIVATION_STORAGE].size
        self.segments[SegmentType.TENSOR].base = self.segments[SegmentType.GRAD_STORAGE].base + \
                                                                self.segments[SegmentType.GRAD_STORAGE].size
        self.segments[SegmentType.OPERATOR].base = self.segments[SegmentType.TENSOR].base + \
                                                                self.segments[SegmentType.TENSOR].size
        self.segments[SegmentType.NET].base = self.segments[SegmentType.OPERATOR].base + \
                                                                self.segments[SegmentType.OPERATOR].size
    
    