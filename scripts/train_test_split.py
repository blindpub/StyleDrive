import yaml
import os

# 打开并读取 yaml 文件
# ori_yaml_path = '/data_storage/haoruiyang/projects/mme2e/StyleDrive/DiffusionDrive/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml'
ori_yaml_path = '/data_storage/haoruiyang/projects/mme2e/GoalFlow/navsim/planning/script/config/common/scene_filter/navtrain.yaml'
ori_yaml_path = '/data_storage/haoruiyang/projects/mme2e/StyleDrive/DiffusionDrive/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml'
ori_yaml_path = '/data_storage/haoruiyang/projects/mme2e/GoalFlow/navsim/planning/script/config/common/scene_filter/navtest.yaml'
with open(ori_yaml_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

token_list = []
video_path_root = "/data_storage/haoruiyang/projects/mme2e/VideoLLaMA3/datasets/videos/styletrain"
video_path_root = "/data_storage/haoruiyang/projects/mme2e/VideoLLaMA3/datasets/videos/styletest"
for token in config['tokens']:
    video_path = os.path.join(video_path_root, token+'.mp4')
    if os.path.exists(video_path):
        token_list.append(token)

config['tokens'] = token_list

# new_yaml_path = '/data_storage/haoruiyang/projects/mme2e/StyleDrive/DiffusionDrive/navsim/planning/script/config/common/train_test_split/scene_filter/styletrain.yaml'
new_yaml_path = '/data_storage/haoruiyang/projects/mme2e/GoalFlow/navsim/planning/script/config/common/scene_filter/styletrain.yaml'
new_yaml_path = '/data_storage/haoruiyang/projects/mme2e/StyleDrive/DiffusionDrive/navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml'
new_yaml_path = '/data_storage/haoruiyang/projects/mme2e/GoalFlow/navsim/planning/script/config/common/scene_filter/styletest.yaml'
with open(new_yaml_path, 'w', encoding='utf-8') as file:
    yaml.dump(config, file, allow_unicode=True)