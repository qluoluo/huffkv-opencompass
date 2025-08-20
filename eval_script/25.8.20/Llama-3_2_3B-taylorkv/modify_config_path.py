import os

dirname = os.path.dirname(__file__)

dest_model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
file_names = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]

opencompass_config_path_list = [os.path.join(dirname, f) for f in file_names if f.startswith('opencompass-eval')]
for opencompass_config_path in opencompass_config_path_list:
    opencompass_config_lines = open(opencompass_config_path).readlines()
    for i, line in enumerate(opencompass_config_lines):
        if line.startswith('model_path = '):
            opencompass_config_lines[i] = f'model_path = "{dest_model_path}"\n'
        
        if line.startswith('ckpt_root = '):
            opencompass_config_lines[i] = f'ckpt_root = "{os.path.join(dirname, "saves")}"\n'

    with open(opencompass_config_path, 'w') as f:
        f.writelines(opencompass_config_lines)


print()
print(f"basename = {os.path.basename(dest_model_path)}")
print()
print(f"lsa-{os.path.basename(dirname)}")
print()
# print(f"bash {os.path.join(dirname, 'train-eval.sh')}")
# print()
print(f"bash {os.path.join(dirname, 'eval-base.sh')}")
print()