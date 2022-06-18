import collections
from queue import Queue
from simulator.memory import Memory
import torch
class Net:
    def __init__(self):
        self.ops = set()
        self.hash = {}
        self.first_op = None
    
    def add_operator(self,*ops):
        """添加算子到网络中
        """
        for op in ops:
            if op:
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

    def get_operator(self,name):
        return self.hash[name]
    
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

    def all_tensors(self,no_duplicate=False):
        """遍历神经网络的所有张量
        """
        visit = set()
        for op in self.topo():
            for tensor_name,tensor in op.get_tensors().tensors.items():
                if no_duplicate:
                    if tensor in visit:
                        continue
                    visit.add(tensor)
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

    def _sim_run_prepare(self,input):
        #设置好输入张量
        Memory().set(self.first_op.tensors.get("output").addr,input)
        #设置好权重张量
        visit_tensor = set()
        for op,tensor_name,tensor in self.all_tensors():
            if tensor in visit_tensor:
                continue
            visit_tensor.add(tensor)
            if tensor.storage.data is not None:
                Memory().set(tensor.addr,torch.from_numpy(tensor.storage.data))

    def sim_run(self,input):
        #准备张量
        self._sim_run_prepare(input)
        #执行
        for op in self.topo():
            op.sim_run()
            
    def sim_run_to(self,input,op_name):
        #准备张量
        self._sim_run_prepare(input)
        #执行
        for op in self.topo():
            op.sim_run()
            if op.name==op_name:
                break
        return op

    def statistic_op(self):
        print("-------- Operator statistic --------")
        table = {}
        for op in self.topo():
            class_name = type(op).__name__
            if class_name in table:
                table[class_name] += 1
            else:
                table[class_name] = 1
        for name,count in table.items():
            print(f"{name}: {count}")
        print("----------------------------------")
        return table

    def deal_input_data(self):
        self.input = self.get_operator("FEdge_0").tensors.get("output")
        self.label = self.get_operator("FEntropy_0").tensors.get("label")
    
    def apply(self,fn):
        for op in self.topo():
            fn(op)