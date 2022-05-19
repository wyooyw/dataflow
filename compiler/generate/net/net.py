import collections
from queue import Queue
class Net:
    def __init__(self):
        self.ops = set()
        self.hash = {}
        self.first_op = None
    
    def add_operator(self,*ops):
        """添加算子到网络中
        """
        for op in ops:
            if self.first_op==None:
                self.first_op = op
            self.ops.add(op)
            self.hash[op.name] = op
            op.net = self
    
    def remove_operator(self,*ops):
        """从网络中删除算子
        """
        for op in ops:
            self.ops.discard(op)
            self.hash[op.name] = None
            op.net = None
    
    def topo(self):
        """按照拓扑序遍历节点
        """
        queue = Queue()
        queue.put(self.first_op)
        record = {}
        
        while not queue.empty():
            op = queue.get()
            if len(op.predecessor)>1:
                if op not in record:
                    record[op] = len(op.predecessor) - 1
                    continue
                elif record[op]>1:
                    record[op] -= 1
                    continue

            for suc in op.successor:
                queue.put(suc)
            
            yield op
    
    def dfs(self):
        """按照深度优先顺序遍历节点
        """
        pass

    def all_tensors(self):
        """遍历神经网络的所有张量
        """
        for op in self.topo():
            for tensor_name,tensor in op.get_tensors().tensors.items():
                yield op,tensor_name,tensor

    def reduce_tensor(self):
        """将指向同一storage的tensor合并,可减少片上内存
        TODO:后续应改为,属性相同的tensor合并,以适应更复杂的情况
        """
        collect = collections.OrderedDict()
        for op,tensor_name,tensor in self.all_tensors():
            storage = tensor.storage
            if storage in collect:
                king_tensor = collect[storage]
                op.tensors.set(tensor_name,king_tensor)
            else:
                collect[storage] = tensor

    def count(self):
        """计算storage和tensor的数量
        """
        visit_storage = set()
        visit_tensor = set()
        stats_tensor = {}
        stats_storage = {}
        for op,tensor_name,tensor in self.all_tensors():
            storage = tensor.storage
            if tensor not in visit_tensor:
                visit_tensor.add(tensor)
                if storage.type not in stats_tensor:
                    stats_tensor[storage.type] = 1
                else:
                    stats_tensor[storage.type] += 1
            if storage not in visit_storage:
                visit_storage.add(storage)
                if storage.type not in stats_storage:
                    stats_storage[storage.type] = 1
                else:
                    stats_storage[storage.type] += 1
            
        return {"tensor":stats_tensor,"storage":stats_storage}

    def set_tensor_index(self):
        """为张量赋予index属性
        """
        index = 0
        visit = set()
        for op,tensor_name,tensor in self.all_tensors():
            if tensor not in visit:
                tensor.index = index
                index += 1
                visit.add(tensor)

        
    
    def __str__(self):
        """将网络转换为字符串,按照算子的拓扑顺序
        """
        strs = []
        for op in self.topo():
            strs.append(str(op))
        return "\n".join(strs)
        # strs = []
        # queue = Queue()
        # queue.put(self.first_op)
        # record = {}
        
        # while not queue.empty():
        #     op = queue.get()
        #     if len(op.predecessor)>1:
        #         if op not in record:
        #             record[op] = len(op.predecessor) - 1
        #             continue
        #         elif record[op]>1:
        #             record[op] -= 1
        #             continue
        #     strs.append(str(op))
        #     for suc in op.successor:
        #         queue.put(suc)
        # return "\n".join(strs)
        # return "\n".join([str(op) for op in self.ops])
