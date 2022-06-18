from executer.execute_functions import bind_table

def warn(s):
    print(f"[Executer][WARN]: {s}")

class Executer(object):
    def __init__(self,net):
        self.net = net

    def _prepare_tensors(self,input,label):
        assert input.ndim==4,"Input should be a 4 dimension tensor."
        assert label.ndim==2,"Label should be a 2 dimension tensor."
        assert input.shape[0]==label.shape[0],"Batch of input and label are not equal!"
        # self.net.first_op.tensors.set_data("output",input)
        # self.net.get_operator("FEntropy_0").set_data("label",label)
        self.net.input.storage.data = input
        self.net.label.storage.data = label

    def execute(self,input,label,to=None):
        self._prepare_tensors(input,label)
        for op in self.net.topo():
            class_name = str(type(op).__name__)
            if class_name in bind_table:
                execute_fn = bind_table[class_name]
                execute_fn(op)
            else:
                warn(f"Operator {op.name} ({class_name}) is not registed in executer.")

            if op.name==to:
                break
        return op
 
 
# @attrs(versionadded="2.2",
#        author="Guido van Rossum")
# def mymethod(f):
#     print(getattr(mymethod,'versionadded',0))
#     print(getattr(mymethod,'author',0))
#     print(f)
 
# if __name__ == "__main__":
#     mymethod(2)