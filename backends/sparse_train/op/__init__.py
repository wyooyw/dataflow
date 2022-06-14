from backends.sparse_train.op.conv_bn import ForwardConvBn
from backends.sparse_train.op.conv_bn_relu import ForwardConvBnRelu
from backends.sparse_train.op.conv_bn_add_relu import ForwardConvBnAddRelu
from backends.sparse_train.op.conv_bn_add_relu_maxpool import ForwardConvBnAddReluMaxpool
from backends.sparse_train.op.back_conv_relu_bn import BackwardConvReluBn
from backends.sparse_train.op.back_conv_split_relu_bn import BackwardConvSplitReluBn
from backends.sparse_train.op.conv_relu_bn_bn import ForwardConvReluBnBn
from backends.sparse_train.op.back_conv_split_relu_bn_bn import BackwardConvSplitReluBnBn
from backends.sparse_train.op.back_conv import BackwardSTConv
from backends.sparse_train.op.linear_softmax_entropy import ForwardLinearSoftmaxEntropy

#AlexNet中用到的
from backends.sparse_train.op.conv_relu import ForwardConvRelu
from backends.sparse_train.op.conv_relu_maxpool import ForwardConvReluMaxpool
from backends.sparse_train.op.linear_relu import ForwardLinearRelu
from backends.sparse_train.op.linear_relu_dropout import ForwardLinearReluDropout
from backends.sparse_train.op.back_conv_relu import BackwardConvRelu
from backends.sparse_train.op.back_conv_maxpool_relu import BackwardConvMaxpoolRelu
from backends.sparse_train.op.back_linear_relu import BackwardLinearRelu
from backends.sparse_train.op.back_linear_dropout_relu import BackwardLinearDropoutRelu