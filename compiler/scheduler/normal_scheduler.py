from compiler.scheduler.scheduler import Scheduler
from queue import Queue
from compiler.graph_ir import OperatorType
class NormalScheduler(Scheduler):
    """参数更新放最后
    """
    def __init__(self):
        super().__init__()
        pass
    
    def schedule(self,net):
        """按照拓扑序遍历节点
        """
        queue = Queue()
        queue.put(net.first_op)
        record = {}
        wu_sequence = []
        op_sequence = []
        
        while not queue.empty():
            op = queue.get()
            if len(op.predecessor)>1:
                if op not in record:
                    record[op] = len(op.predecessor) - 1
                    continue
                elif record[op]>1:
                    record[op] -= 1
                    continue
            wu = []
            wg = []
            bw = []
            fw = []
            for suc in op.successor:
                if suc.type==OperatorType.FORWARD:
                    fw.append(suc)
                elif suc.type==OperatorType.BACKWARD:
                    bw.append(suc)
                elif suc.type==OperatorType.WEIGHT_GRADIENT:
                    wg.append(suc)
                elif suc.type==OperatorType.WEIGHT_UPDATE:
                    wu.append(suc)
                else:
                    assert False,f"Unknown op type: {suc.type}"
            for suc in wu:
                queue.put(suc)
            for suc in bw:
                queue.put(suc)
            for suc in wg:
                queue.put(suc)
            for suc in fw:
                queue.put(suc)
            
            if op.type==OperatorType.WEIGHT_UPDATE:
                wu_sequence.append(op)
            else:
                op_sequence.append(op)
        op_sequence.extend(wu_sequence)
        last = None
        for op in op_sequence:
            op.remove_all_predecessor()
            op.remove_all_successor()
            if last==None:
                last = op
            else:
                op.connect_predecessor(last)
                last = op
        
