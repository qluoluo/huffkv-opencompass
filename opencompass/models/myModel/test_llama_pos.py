from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
)

model(0)