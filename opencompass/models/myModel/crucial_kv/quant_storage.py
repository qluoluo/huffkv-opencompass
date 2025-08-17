import torch
from typing import Optional, List
from functools import partial

# from .huffkv_quanthuff_utils import quanthuff_compress, quanthuff_decompress
from .quant_utils import quantize, dequantize


class QuantStorage:
    """
    分层量化数据存储管理器

    功能：
    1. 第一次append：数据会被量化存储（prefill阶段）
    2. 后续append：保持未量化数据，避免频繁量化/反量化
    3. 返回反量化后的完整数据
    """

    def __init__(
        self,
        nbits: int,
        quant_dim: int,
        name: str,
        debug: bool,
        minus_mean: bool = True,
    ):
        """
        初始化量化存储器

        Args:
            nbits: 量化位数
            quant_dim: 量化维度
        """
        self.nbits = nbits
        self.quant_dim = quant_dim
        self.name = name
        self.debug = debug

        # print(f"### init quantstorage {name}")

        # 初始化量化和反量化函数
        self.compress_func = partial(quantize, nbits=nbits, dim=quant_dim)
        self.decompress_func = partial(dequantize)

        # self.compress_func = partial(quanthuff_compress, nbits=nbits, dim=quant_dim)
        # self.decompress_func = partial(quanthuff_decompress)

        # prefill阶段的量化数据存储（第一次append的数据）
        self._quantized_prefill_data = None

        # append阶段的未量化数据存储（后续append的数据）
        self._decode_data = None

        # 减去每个channel的bias
        self.minus_mean = minus_mean
        self.prefill_data_mean = None

        self.cache_length = 0

    def append(self, new_data: torch.Tensor) -> None:
        """
        添加新数据
        第一次调用时数据会被量化存储，后续调用保持未量化

        Args:
            new_data: 新的数据张量
        """
        self.cache_length += new_data.shape[-2]

        if self.minus_mean:
            if self.prefill_data_mean is None:
                self.prefill_data_mean = new_data.mean(dim=-2, keepdim=True)
            new_data = new_data - self.prefill_data_mean

        if self._quantized_prefill_data is None:
            # 第一次append，量化存储（prefill阶段）
            if self.debug:
                print(f"QuantStorage append prefill data, now {self.get_length()=}")
            self._quantized_prefill_data = self.compress_func(new_data)
        else:
            # 后续append，保持未量化
            if self.debug:
                print(f"QuantStorage append decode data, now {self.get_length()=}")
            if self._decode_data is None:
                self._decode_data = new_data
            else:
                self._decode_data = torch.cat([self._decode_data, new_data], dim=-2)

    def get_data(self) -> Optional[torch.Tensor]:
        """
        获取反量化后的完整数据

        Returns:
            反量化后的完整张量，如果存储为空则返回None
        """
        ret_data_list = []

        # 添加prefill部分（需要反量化）
        # if self._quantized_prefill_data is not None:
        #     prefill_dequantized = self.decompress_func(*self._quantized_prefill_data)
        #     ret_data_list.append(prefill_dequantized)

        # 添加append部分（已经是未量化的）
        if self._decode_data is not None:
            ret_data_list.append(self._decode_data)

        if len(ret_data_list) == 0:
            return None
        
        ret_data = torch.cat(ret_data_list, dim=-2)

        if self.minus_mean:
            # prefill_dequantized = prefill_dequantized + self.prefill_data_mean
            ret_data = ret_data + self.prefill_data_mean

        return ret_data

    def get_length(self) -> int:
        """
        获取存储数据的长度

        Returns:
            存储数据的长度
        """
        return self.cache_length
        # ret_length = 0

        # if self._quantized_prefill_data is not None:
        #     ret_length += self._quantized_prefill_data[0].shape[-2]

        # if self._decode_data is not None:
        #     ret_length += self._decode_data.shape[-2]

        # return ret_length
