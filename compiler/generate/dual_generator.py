class DualGenerator:
    def __init__(self):
        pass

    def getForwardOp(self):
        return self.forward_op

    def getBackwardOp(self):
        return self.backward_op

    def getDual(self):
        return self.forward_op,self.backward_op