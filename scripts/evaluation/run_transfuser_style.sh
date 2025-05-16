TRAIN_TEST_SPLIT=styletest
CKPT=$NAVSIM_EXP_ROOT/training_transfuser_style_agent/2025.05.15.03.21.40/lightning_logs/version_0/checkpoints/final.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=transfuser_style_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=eval_transfuser_style_agent
