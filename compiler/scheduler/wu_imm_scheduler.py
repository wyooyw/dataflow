from compiler.scheduler.scheduler import Scheduler
from queue import Queue
from compiler.graph_ir import OperatorType
class WUImmScheduler(Scheduler):
    """参数立即更新
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
        op_sequence = []
        wu_queue = Queue()
        while not (queue.empty() and wu_queue.empty()):
            if not wu_queue.empty():
                op = wu_queue.get()
            else:
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
                wu_queue.put(suc)
            for suc in bw:
                queue.put(suc)
            for suc in wg:
                queue.put(suc)
            for suc in fw:
                queue.put(suc)
            op_sequence.append(op)
        last = None
        for op in op_sequence:
            op.remove_all_predecessor()
            op.remove_all_successor()
            if last==None:
                last = op
            else:
                op.connect_predecessor(last)
                last = op
        
