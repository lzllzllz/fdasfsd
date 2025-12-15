'''
# run
python run.py ./results/simulation/configs/params_all.json 1> run.log 2>&1 
'''
import json 
import sys
import os
import numpy as np, pandas as pd
from subprocess import call

    
def load_config(cfg_file):##函数负责加载JSON格式的配置文件，并将其转换为配置列表。
    with open(cfg_file) as f:
        params_all = json.load(f)
    cfgs = list(params_all.values())
    return cfgs

def cfg_string(cfg):##函数将配置字典转换为字符串表示形式，便于比较和存储。
    ks = sorted(cfg.keys())
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()

def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs

def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs

def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)

def run(cfg_file):
    configs = load_config(cfg_file)

    used_cfg_file = os.path.join(configs[0]['output_dir'], 'configs', 'used_configs.txt' )

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    for i in range(len(configs)):
        cfg = configs[i]

        ##遍历每个配置，跳过已经使用过的
        if is_used_cfg(cfg, used_cfg_file):
            print('Configuration used, skipping')
            continue

        save_used_cfg(cfg, used_cfg_file)

        ###开始训练，打印配置信息
        print('------------------------------')
        print('\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.items()]))

        flags = ' '.join('-%s %s' % (k,str(v)) for k,v in cfg.items())


        print('\ncall: ', 'python train.py %s' % flags)
        call('python train.py %s' % flags, shell=True)###就是相当于train.py 后面接的是flags



if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print('Usage: python cfr_param_search.py <config file>')
        print('use default path: ./results/simulation/configs/params_all.json')
        run('./results/simulation/configs/params_all.json')
    else:
        run(sys.argv[1])

