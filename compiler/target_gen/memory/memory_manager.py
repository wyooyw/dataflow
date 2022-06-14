from compiler.utils.singleton import singleton
from compiler.utils.rectangle import Rectangle,RectangleManager,getRectangleManager
from compiler.target_gen.memory.storage import Storage,StorageType
from compiler.target_gen.memory.tensor import Tensor

from functools import reduce
import collections

@singleton
class MemoryManager(object):
    def __init__(self):
        # self.net = Net()
        self.tensor_list = []
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
                    type,
                    content=None):
        size = reduce(lambda x, y: x*y, shape)
        storage = Storage(size=size,content=content,type=type)
        tensor = Tensor(storage=storage,shape=shape)
        self.tensor_list.append(tensor)

        return tensor

    def allocWeight(self,shape,content=None):
        return self.allocTensor(shape=shape,type=StorageType.WEIGHT,content=content)
    
    def allocActivation(self,shape,content=None):
        return self.allocTensor(shape=shape,type=StorageType.ACTIVATION,content=content)
    
    def allocWeightGrad(self,shape,content=None):
        return self.allocTensor(shape=shape,type=StorageType.WEIGHT_GRAD,content=content)

    def allocFeatureGrad(self,shape,content=None):
        return self.allocTensor(shape=shape,type=StorageType.FEATURE_GRAD,content=content)
    
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
    
    def tensor_memory_layout2(self,net,show_image=False,save_image=False):
        """利用求解二维装箱问题来进行内存布局
        """

        #为每个张量设定生命周期
        self._mark_life_time(net)

        rec_manager = getRectangleManager()
        visit_tensor = set()
        addr = 0
        for op,tensor_name,tensor in net.all_tensors():
            if tensor in visit_tensor:
                continue
            if tensor.storage.type==StorageType.WEIGHT:
                tensor.addr = addr
                size = tensor.storage.size * 2
                addr += size
                visit_tensor.add(tensor)
                continue

            rec = self._make_rectangle(tensor)
            visit_tensor.add(tensor)
            tensor.rec = rec
            rec_manager.add_rectangle(rec)
            
        memory_max = rec_manager.layout()
        print("memory_max:",memory_max)
        print("weight_size:",addr)
        if show_image:
            rec_manager.animate()
            # rec_manager.paint()
        if save_image:
            rec_manager.save()
        max_addr = 0
        for tensor in visit_tensor:
            if hasattr(tensor,"rec"):
                tensor.addr = tensor.rec.y_range[0] + addr
                max_addr = max(max_addr,tensor.addr+tensor.storage.size*2)
        print(max_addr,"B")

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
    
    def count_read_and_write_times(self,net):
        read_count = 0
        write_count = 0
        cur_op = None
        cur_read_count = 0
        cur_write_count = 0
        print("{}{}{}".format("op".ljust(20),
                                "read_count".ljust(20),
                                "write_count".ljust(20)))
        for op,tensor_name,tensor in net.all_tensors():
            if op.name=="FEdge_0" or op.name=="BEdge_0":
                continue
            if not op==cur_op:
                if not cur_op==None:
                    print("{}{}{}".format(cur_op.name.ljust(20),
                                str(cur_read_count).ljust(20),
                                str(cur_write_count).ljust(20)))
                cur_op = op
                cur_read_count = 0
                cur_write_count = 0
            if tensor_name in op.tensors.read_tensors:
                read_count += tensor.storage.size
                cur_read_count += tensor.storage.size
            elif tensor_name in op.tensors.write_tensors:
                write_count += tensor.storage.size
                cur_write_count += tensor.storage.size
            else:
                pass
                #assert False,f"A tensor not is not be read or write!{op.name},{tensor_name}"
        print("{}{}{}".format(cur_op.name.ljust(20),
                                str(cur_read_count).ljust(20),
                                str(cur_write_count).ljust(20)))
        print(f"read_count:{read_count},write_count:{write_count}")