import os, yaml
import json
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
config_path = osp.join(dirname, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
save_path = config['dpo_sampling']['save_path']

alldata = []
for i in range(30):
    filename_json = f'temp/final_res_{i}k.json'
    if os.path.isfile(filename_json):
        with open(filename_json, 'r') as file:
            data = json.load(file)
        alldata.extend(data)
with open(save_path, 'w') as output_file:
    json.dump(alldata, output_file, ensure_ascii=False, indent=4)
