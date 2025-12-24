from .accessory import LLaMA2AccessoryModel  # noqa: F401
from .ai360_api import AI360GPT  # noqa: F401
from .alaya import AlayaLM  # noqa: F401
from .baichuan_api import BaiChuan  # noqa: F401
from .baidu_api import ERNIEBot  # noqa: F401
from .bailing_api_oc import BailingAPI  # noqa: F401
from .base import BaseModel, LMTemplateParser  # noqa: F401
from .base_api import APITemplateParser, BaseAPIModel  # noqa: F401
from .bluelm_api import BlueLMAPI  # noqa: F401
from .bytedance_api import ByteDance  # noqa: F401
from .claude_allesapin import ClaudeAllesAPIN  # noqa: F401
from .claude_api import Claude  # noqa: F401
from .claude_sdk_api import ClaudeSDK  # noqa: F401
from .deepseek_api import DeepseekAPI  # noqa: F401
from .doubao_api import Doubao  # noqa: F401
from .gemini_api import Gemini  # noqa: F401
from .glm import GLM130B  # noqa: F401
from .huggingface import HuggingFace  # noqa: F401
from .huggingface import HuggingFaceCausalLM  # noqa: F401
from .huggingface import HuggingFaceChatGLM3  # noqa: F401
from .huggingface_above_v4_33 import HuggingFaceBaseModel  # noqa: F401
from .huggingface_above_v4_33 import HuggingFacewithChatTemplate  # noqa: F401
from .hunyuan_api import Hunyuan  # noqa: F401
from .intern_model import InternLM  # noqa: F401
from .interntrain import InternTrain  # noqa: F401
from .krgpt_api import KrGPT  # noqa: F401
from .lightllm_api import LightllmAPI, LightllmChatAPI  # noqa: F401
from .llama2 import Llama2, Llama2Chat  # noqa: F401
from .minimax_api import MiniMax, MiniMaxChatCompletionV2  # noqa: F401
from .mistral_api import Mistral  # noqa: F401
from .mixtral import Mixtral  # noqa: F401
from .modelscope import ModelScope, ModelScopeCausalLM  # noqa: F401
from .moonshot_api import MoonShot  # noqa: F401
from .nanbeige_api import Nanbeige  # noqa: F401
from .openai_api import OpenAI  # noqa: F401
from .openai_api import OpenAISDK  # noqa: F401
from .openai_streaming import OpenAISDKStreaming  # noqa: F401
from .pangu_api import PanGu  # noqa: F401
from .qwen_api import Qwen  # noqa: F401
from .rendu_api import Rendu  # noqa: F401
from .sensetime_api import SenseTime  # noqa: F401
from .stepfun_api import StepFun  # noqa: F401
from .turbomind import TurboMindModel  # noqa: F401
from .turbomind_with_tf_above_v4_33 import TurboMindModelwithChatTemplate  # noqa: F401
from .unigpt_api import UniGPT  # noqa: F401
from .vllm import VLLM  # noqa: F401
from .vllm_with_tf_above_v4_33 import VLLMwithChatTemplate  # noqa: F401
from .xunfei_api import XunFei, XunFeiSpark  # noqa: F401
from .yayi_api import Yayi  # noqa: F401
from .yi_api import YiAPI  # noqa: F401
from .zhipuai_api import ZhiPuAI  # noqa: F401
from .zhipuai_v2_api import ZhiPuV2AI  # noqa: F401


from .myModel.hf_strip_model import HuggingFaceCausalLM_Strip
from .myModel.hf_niah_model import HuggingFaceCausalLMForNIAH

try:
    from .myModel.general_quant.general_quant_model import (
        LlamaForCausalLM_GeneralQuant_OC,
    )
    from .myModel.general_quant_debug.general_quant_model import (
        LlamaForCausalLM_GeneralQuant_OC as LlamaForCausalLM_GeneralQuant_Debug_OC,
    )
    from .myModel.general_quant_v0816.quant_model import LlamaForCausalLM_GQ_V0816_OC
except:
    pass

try:
    from .myModel.kivi.kivi_model import LlamaForCausalLM_KIVI_OC
except:
    pass

try:
    from .myModel.huffkv.huffkv_quant_model import LlamaForCausalLM_HuffKV_OC
    from .myModel.huffkv_8_5.huffkv_quant_model import (
        LlamaForCausalLM_HuffKV_OC as LlamaForCausalLM_HuffKV_8_5_OC,
    )
except:
    pass


try:
    from .myModel.simpleprefill_quant.simpleprefill_quant_model import (
        LlamaForCausalLM_SimplePrefill_OC,
    )
except:
    pass

try:
    from .myModel.crucial_kv.quant_model import LlamaForCausalLM_CrucialKV_OC
except:
    pass

try:
    from .myModel.taylor_kv.oc_model import LlamaForCausalLM_TaylorKV_OC
    from .myModel.simple_taylorkv.oc_model import LlamaForCausalLM_Simple_TaylorKV_OC
except:
    pass

try:
    from .myModel.bucket_attn.oc_model import LlamaForCausalLM_BucketAttn_OC
except:
    pass

try:
    from .myModel.interkv.oc_model import LlamaForCausalLM_InterKV_OC
except:
    pass


try:
    from .myModel.ffa.oc_model import HF_ForCausalLM_FFA_OC
    print("[IMPORT] FFA import sucess!")
except:
    pass

try:
    from .myModel.quest.oc_model import LlamaForCausalLM_Quest_OC
    print("[IMPORT] Quest import success!")
except:
    pass

try:
    from .myModel.twilight.oc_model import LlamaForCausalLM_Twilight_OC
    print("[IMPORT] Twilight import success!")
except:
    pass
