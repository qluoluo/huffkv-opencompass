import torch
from typing import Optional, List, Tuple
from functools import partial
from .huffkv_quanthuff_utils import quanthuff_compress, quanthuff_decompress
from .huffkv_quant_utils import quantize, dequantize

class QuantStorage:
    
    def __init__(
            self, 
            nbits: int,
            channel_group_size: int,
            token_group_size: int,
            name: str = "QuantStorage",
            debug: bool = False,
            minus_mean: bool = True
        ):
        self.nbits = nbits
        # 初始化实际分组大小变量（后续会调整）
        self.actual_channel_group_size = channel_group_size
        self.actual_token_group_size = token_group_size
        self.name = name
        self.debug = debug
        self.minus_mean = minus_mean
        
        # 初始化量化和反量化函数
        self.compress_func = partial(quantize, nbits=nbits, dim=-1)
        self.decompress_func = partial(dequantize, dim=-1)
        
        self._prefill_main_quantized = None
        self._prefill_remain = None
        self._decode_data = None
        self.prefill_data_mean = None
        self.original_shape = None
        
    def append(self, new_data: torch.Tensor) -> None:
        if self.minus_mean:
            if self.prefill_data_mean is None:
                self.prefill_data_mean = new_data.mean(dim=-2, keepdim=True)
            new_data = new_data - self.prefill_data_mean

        if self._prefill_main_quantized is None:
            # 确定实际分组大小（处理-1情况）
            if self.actual_channel_group_size == -1:
                self.actual_channel_group_size = new_data.shape[-1]
            if self.actual_token_group_size == -1:
                self.actual_token_group_size = new_data.shape[-2]
                
            # 验证分组大小有效性
            if self.actual_channel_group_size <= 0:
                raise ValueError(f"Invalid channel_group_size: {self.actual_channel_group_size}")
            if self.actual_token_group_size <= 0:
                raise ValueError(f"Invalid token_group_size: {self.actual_token_group_size}")
                
            # 检查维度可整除性
            if new_data.shape[-1] % self.actual_channel_group_size != 0:
                raise ValueError(
                    f"Channel dimension {new_data.shape[-1]} must be divisible by "
                    f"channel_group_size {self.actual_channel_group_size}"
                )
                
            self.original_shape = new_data.shape[:-2]
            num_tokens = new_data.shape[-2]
            channels = new_data.shape[-1]
            
            # 计算主token数量（可被分组整除的部分）
            main_tokens = (num_tokens // self.actual_token_group_size) * self.actual_token_group_size
            remain_tokens = num_tokens - main_tokens
            
            # 处理可分组部分
            if main_tokens > 0:
                main_data = new_data[..., :main_tokens, :]
                
                # 重塑张量：[...][主token组数][token组大小][通道组数][通道组大小]
                main_data = main_data.reshape(
                    *self.original_shape, 
                    main_tokens // self.actual_token_group_size, 
                    self.actual_token_group_size, 
                    channels
                )
                main_data = main_data.reshape(
                    *self.original_shape,
                    main_tokens // self.actual_token_group_size,
                    self.actual_token_group_size,
                    channels // self.actual_channel_group_size,
                    self.actual_channel_group_size
                )
                
                # 展平分组维度：[...][总组数][token组大小×通道组大小]
                main_data = main_data.permute(
                    *range(len(self.original_shape)),
                    -4, -2, -3, -1
                ).flatten(-3, -2).flatten(-2, -1)
                
                # 量化数据
                self._prefill_main_quantized = self.compress_func(main_data)
            
            # 处理剩余token
            if remain_tokens > 0:
                self._prefill_remain = new_data[..., main_tokens:, :].clone()
            else:
                self._prefill_remain = None
        else:
            # 后续追加数据（直接存储）
            if self._decode_data is None:
                self._decode_data = new_data
            else:
                self._decode_data = torch.cat([self._decode_data, new_data], dim=-2)
    
    def get_data(self) -> Optional[torch.Tensor]:
        if self._prefill_main_quantized is None:
            return None
        
        # 反量化主数据
        if self._prefill_main_quantized is not None:
            main_data = self.decompress_func(*self._prefill_main_quantized)
            
            # 计算原始形状参数
            num_token_groups = main_data.shape[-2] // (self.original_shape[-1] // self.actual_channel_group_size)
            num_channel_groups = self.original_shape[-1] // self.actual_channel_group_size
            
            # 恢复分组维度：[...][主token组数][通道组数][token组大小][通道组大小]
            restored_shape = (
                *self.original_shape,
                num_token_groups,
                num_channel_groups,
                self.actual_token_group_size,
                self.actual_channel_group_size
            )
            main_data = main_data.reshape(restored_shape)
            
            # 恢复原始维度：[...][主token数][通道数]
            main_data = main_data.permute(
                *range(len(self.original_shape)),
                -4, -2, -3, -1
            ).reshape(
                *self.original_shape,
                num_token_groups * self.actual_token_group_size,
                self.original_shape[-1]
            )
            
            # 添加剩余token
            if self._prefill_remain is not None:
                main_data = torch.cat([main_data, self._prefill_remain], dim=-2)
        else:
            main_data = self._prefill_remain if self._prefill_remain is not None else torch.empty(
                *self.original_shape, 0, self.original_shape[-1],
                device=self._prefill_remain.device if self._prefill_remain is not None else 'cpu'
            )
        
        # 添加后续追加的数据
        if self._decode_data is not None:
            main_data = torch.cat([main_data, self._decode_data], dim=-2)
        
        # 恢复均值
        if self.minus_mean and self.prefill_data_mean is not None:
            main_data = main_data + self.prefill_data_mean
        
        return main_data

    def get_length(self) -> int:
        ret_length = 0
        if self._prefill_main_quantized is not None:
            # 计算主token数量
            num_groups = self._prefill_main_quantized[0].shape[-2]
            ret_length += num_groups * self.actual_token_group_size
        if self._prefill_remain is not None:
            ret_length += self._prefill_remain.shape[-2]
        if self._decode_data is not None:
            ret_length += self._decode_data.shape[-2]
        return ret_length