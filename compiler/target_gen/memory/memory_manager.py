from compiler.utils.singleton import singleton
from compiler.utils.rectangle import Rectangle,RectangleManager,getRectangleManager
from compiler.target_gen.memory.net import Net
from compiler.target_gen.memory.mem_operator import MemOperator
from compiler.target_gen.memory.storage import Storage,StorageType
from compiler.target_gen.memory.tensor import Tensor
from compiler.target_gen.memory.segment import Segment,SegmentType

from functools import reduce
import collections

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
        self.alloc_num = 0
        pass

    def get_tensor_num(self):
        return len(self.tensor_list)
    
    def get_tensor_size(self):
        size = reduce(lambda x,y:x+y,[reduce(lambda x,y:x*y, tensor.shape) for tensor in self.tensor_list])
        return size

    def get_tensor_max(self):
        sizes = [reduce(lambda x,y:x*y, tensor.shape) for tensor in self.tensor_list]
        return max(sizes)
    
    def allocTensor(self,
                    shape,
                    storage_segment:Segment,
                    type,
                    content=None):
        size = reduce(lambda x, y: x*y, shape)
        storage = Storage(size=size,content=content,type=type)
        tensor = Tensor(storage=storage,shape=shape)
        self.tensor_list.append(tensor)
        storage_segment.size += size

        tensor_segment = self.segments[SegmentType.TENSOR]
        ndim = len(shape)
        tensor_segment.size += 1 + ndim + 1 #1:ndim; ndim:shape; 1:storage_addr
        return tensor

    def allocWeight(self,shape,content=None):
        segment = self.segments[SegmentType.WEIGHT_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,type=StorageType.WEIGHT,content=content)
    
    def allocActivation(self,shape,content=None):
        segment = self.segments[SegmentType.ACTIVATION_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,type=StorageType.ACTIVATION,content=content)
    
    def allocWeightGrad(self,shape,content=None):
        segment = self.segments[SegmentType.GRAD_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,type=StorageType.WEIGHT_GRAD,content=content)

    def allocFeatureGrad(self,shape,content=None):
        segment = self.segments[SegmentType.GRAD_STORAGE]
        return self.allocTensor(storage_segment=segment,shape=shape,type=StorageType.FEATURE_GRAD,content=content)
    
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
    
    def memory_layout(self,net):
        """内存布局规划
        """
        pass
    
    def tensor_memory_layout(self,net):
        """张量内存布局规划
        这里得到的内存地址是偏移量,真实地址需要加上tensor_base_addr得到
        """
        collect = collections.OrderedDict()
        visit = set()
        for op,tensor_name,tensor in net.all_tensors():
            storage = tensor.storage
            storage_type = storage.type
            if storage_type not in collect:
                collect[storage_type] = []
            if storage not in visit:
                collect[storage_type].append(storage)
                visit.add(storage)
        addr = 0
        bases = collections.OrderedDict()
        for storage_type,storage_list in collect.items():
            print(f"base of {storage_type} is {addr}B = {int(addr/1024*100)/100}KB")
            for storage in storage_list:
                storage.addr = addr
                addr += storage.size*2 #乘2是因为fp16
        
        print(f"end at {addr}B = {int(addr/1024*100)/100}KB")
    
    def tensor_memory_layout2(self,net):
        """利用求解二维装箱问题来进行内存布局
        """

        #为每个张量设定生命周期
        self._mark_life_time(net)

        rec_manager = getRectangleManager()
        visit_tensor = set()
        for op,tensor_name,tensor in net.all_tensors():
            if tensor in visit_tensor:
                continue
            if tensor.storage.type==StorageType.WEIGHT:
                continue

            rec = self._make_rectangle(tensor)
            visit_tensor.add(tensor)
            rec_manager.add_rectangle(rec)
            
        memory_max = rec_manager.layout()
        print("memory_max:",memory_max)
        rec_manager.paint()
        rec_manager.save()

    def _mark_life_time(self,net):
        """为每个张量设定生命周期
        """
        visit_op = set()
        visit_tensor = set()
        time = -1
        for op,tensor_name,tensor in net.all_tensors():
            if op not in visit_op:
                visit_op.add(op)
                time += 1
            if tensor not in visit_tensor:
                visit_tensor.add(tensor)
                tensor.life_begin = time
                tensor.life_end = time+1
            else:
                tensor.life_end = time+1
    
    def _make_rectangle(self,tensor):
        """由tensor构建矩形
        长为tensor生命周期
        高为tensor内存占用大小
        """
        x_range = (tensor.life_begin,tensor.life_end)
        height = tensor.storage.size * 2#fp16
        if tensor.storage.type==StorageType.ACTIVATION:
            color = "blue"
        elif tensor.storage.type==StorageType.FEATURE_GRAD:
            color = "green"
        elif tensor.storage.type==StorageType.WEIGHT_GRAD:
            color = "red"
        else:
            assert False
        return Rectangle(x_range,height,color,tensor)
    