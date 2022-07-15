# 格式转换 进度

## 卷积相关

| 函数名                                     | 说明                             | 是否完成 |
| ------------------------------------------ | -------------------------------- | -------- |
| convert_conv_forward_fm                    | 卷积前传FM                       | 是       |
| convert_conv_forward_stride_2_kernel_1_fm  | 卷积前传FM（stride=2，kernel=1） | 是       |
| convert_conv_forward_stride_2_kernel_3_fm  | 卷积前传FM（stride=2，kernel=3） | 是       |
| convert_conv_forward_wm                    | 卷积前传WM                       | **否**   |
| convert_conv_forward_stride_2_kernel_3_wm  | 卷积前传WM（stride=2，kernel=3） | **否**   |
| convert_conv_backward_fm                   | 卷积反传FM                       | 是       |
| convert_conv_backward_stride_2_kernel_3_fm | 卷积反传FM（stride=2，kernel=3） | 是       |
| convert_conv_backward_wm                   | 卷积反传WM                       | **否**   |
| convert_conv_wg_fm                         | 卷积WG，FM                       | **否**   |
| convert_conv_wg_stride_2_kernel_3_fm       | 卷积WG，FM（stride=2，kernel=3） | **否**   |
| convert_conv_wg_wm                         | 卷积WG，WM                       | **否**   |


## 全连接相关：

| 函数名                     | 说明         | 是否完成 |
| -------------------------- | ------------ | -------- |
| convert_linear_forward_fm  | 全连接前传FM | 是       |
| convert_linear_forward_wm  | 全连接前传WM | **否**   |
| convert_linear_backward_fm | 全连接反传FM | 是       |
| convert_linear_backward_wm | 全连接反传WM | **否**   |
| convert_linear_wg_fm       | 全连接WG,FM  | **否**   |
| convert_linear_wg_wm       | 全连接WG,FM  | **否**   |

