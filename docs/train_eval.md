# StyleDrive Data download & DiffusionDrive Training and Evaluation

## 0. Prepare StyleDrive Data AND CKPT

Please download the StyleDrive data from [huggingface](https://huggingface.co/datasets/Ryhn98/StyleDrive)

If you want to run inference, modify the StyleDrive data path in StyleDrive/navsim/planning/script/config/common/agent. Then run step 1.2 and step 3.

Checkpoints are also provided in [huggingface](https://huggingface.co/datasets/Ryhn98/StyleDrive)

## 1. Cache dataset for faster training and evaluation

### 1.1 cache dataset for training

```bash
# cache dataset for training
python navsim/planning/script/run_dataset_caching.py agent=diffusiondrive_style_agent experiment_name=training_diffusiondrive_style_agent train_test_split=styletrain
```

### 1.2 cache dataset for evaluation

```bash
# cache dataset for evaluation
python navsim/planning/script/run_metric_caching.py train_test_split=styletest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

## 2. Training

If your training machine does not have network access, you should download the pretrained ResNet-34 model from [huggingface](https://huggingface.co/timm/resnet34.a1_in1k) and upload it to your training machine. You should also download the [clustered anchors](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy)

Before starting training, ensure that you correctly set the `bkb_path` to the path of the downloaded pretrained ResNet-34 model. Additionally, set the `plan_anchor_path` to the path of the downloaded clustered anchors in the file located at `/path/to/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py`.

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=diffusiondrive_style_agent \
        experiment_name=training_diffusiondrive_style_agent  \
        train_test_split=styletrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False
```

## 3. Evaluation

You can use the following command to evaluate the trained model `export CKPT=/path/to/your/checkpoint.pth`, for example, you can download provided checkpoint from [huggingface](https://huggingface.co/datasets/Ryhn98/StyleDrive), and set `CKPT=/path/to/downloaded/ckpts`:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=styletest \
        agent=diffusiondrive_style_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        experiment_name=diffusiondrive_style_agent_eval
```
