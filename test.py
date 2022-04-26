from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.target_gen.memory.segment import Segment,SegmentType
from compiler.generate.net.net import Net
from compiler.generate.op.conv import ConvDualGenerator,ForwardConv2d
from compiler.generate.op.relu import ReluDualGenerator,ForwardRelu
from compiler.generate.op.split import SplitDualGenerator
from compiler.generate.op.fuse_conv_relu import ForwardFuseConvRelu

from compiler.generate.op.aggregate import AggregateDualGenerator
from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.merger import Finder
def testMemoryManager():
    mem = MemoryManager()
    mem.calcBases()
    for name,segment in mem.segments.items():
        print(f"name:{name}")
        print(f"\tsegment.base:{segment.base},segment.size:{segment.size}\n")

def testDualGenerator():
    conv_forward,conv_backward = ConvDualGenerator(in_batch=1,
                                                    in_width=4,
                                                    in_height=4,
                                                    in_channels=3,
                                                    out_channels=2,
                                                    kernel_size=2,
                                                    padding=0).getDual()
    relu_forward,relu_backward = ReluDualGenerator(in_batch=1,
                                                    in_width=3,
                                                    in_height=3,
                                                    in_channels=2).getDual()
    conv_forward.next_output_to(relu_forward)
    conv_forward.connect_successor(relu_forward)
    
    conv2_forward,conv2_backward = ConvDualGenerator(in_batch=1,
                                                    in_width=3,
                                                    in_height=3,
                                                    in_channels=2,
                                                    out_channels=2,
                                                    kernel_size=3,
                                                    padding=1).getDual()
    
    relu_forward.next_output_to(conv2_forward)
    relu_forward.connect_successor(conv2_forward)

    relu2_forward,relu2_backward = ReluDualGenerator(in_batch=1,
                                                    in_width=3,
                                                    in_height=3,
                                                    in_channels=2).getDual()
    conv2_forward.next_output_to(relu2_forward)
    conv2_forward.connect_successor(relu2_forward)

    net = Net()
    for op in [conv_forward,relu_forward,conv2_forward,relu2_forward]:
        net.add_operator(op)

    print(net)

    pattern = [ForwardConv2d,ForwardRelu]
    finder = Finder(net,pattern)

    find = finder.find()
    for f in find:
        fuse_conv_relu = ForwardFuseConvRelu.merge(conv=f[0],relu=f[1])
    print("---------------")
    print(net)

def testPointer():
    from compiler.utils.pointer import Pointer
    ptr = Pointer(None)
    ptr2 = ptr
    print(ptr,ptr2)
    ptr.set("helloworld")
    print(ptr,ptr2)

class MyClass:
    def __init__(self):
        # print(unique_class_name(self))
        pass

if __name__=="__main__":
    # testPointer()
    testDualGenerator()
    # testMemoryManager()
    # c1 = MyClass()
    # print(type(c1).__name__)
    # c2 = MyClass()