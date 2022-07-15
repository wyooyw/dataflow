class FrameBuilder(object):
    def __init__(self):
        self.hook_dict = {}
        pass
    
    def set_axis_type(self,axis_type):
        self.axis_type = axis_type

    def add_axis_tail_hook(self,dims,hook):
        self.hook_dict[dims] = 