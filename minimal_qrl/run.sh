#!/bin/bash
# 一键运行最小 QRL 训练脚本
# 支持 Apple Silicon MPS 加速

# 默认simple_grid环境（env-name 默认等于 env-type，可省略）
python minimal_qrl/train.py \
    --device auto \
    --env-type simple_grid \
    --output-dir ./results/minimal_qrl \
    --grid-size 10 10 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 8000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-n-pairs 500 \
    --visualization-interval 1000

"""

# obstacle环境
python minimal_qrl/train.py \
    --device auto \
    --env-type obstacle \
    --output-dir ./results/minimal_qrl \
    --grid-resolution 50 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 8000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-n-pairs 500 \
    --visualization-interval 1000 \
    --planning-num-action-candidates 64 \
    --planning-visualize-failures \
    --planning-visualize-interval 2000

"""

"""

# 在obstacle环境下使用lookahead
python minimal_qrl/train.py \
    --device auto \
    --env-type obstacle \
    --output-dir ./results/minimal_qrl \
    --grid-resolution 100 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 10000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-n-pairs 500 \
    --visualization-interval 1000 \
    --planning-num-action-candidates 64 \
    --planning-execution-modes lookahead \
    --lookahead-horizon 5 \
    --lookahead-num-sequences 64 \
    --lookahead-step-cost-weight 0.0 \
    --lookahead-collision-penalty 0.0 \
    --lookahead-distance-types qrl,euclidean

"""

# 如果需要更快的训练（减少评估和可视化频率）:
# python minimal_qrl/train.py \
#     --device auto \
#     --batch-size 512 \
#     --eval-interval 2000 \
#     --eval-n-pairs 300 \
#     --visualization-interval 2000


"""

# 评估模型使用qrl和euclidean距离的区别（obstacle2d环境）
python minimal_qrl/eval/execution_mode_eval.py \
  --checkpoint results/minimal_qrl/checkpoint_final.pth \
  --output-dir results/minimal_qrl \
  --env-name obstacle2d \
  --max-steps 300 \
  --grid-resolution 120 \
  --n-trials 200 \
  --seed 0 \
  --num-action-candidates 32 \
  --execution-modes lookahead \
  --lookahead-horizon 5 \
  --lookahead-num-sequences 64 \
  --lookahead-step-cost-weight 0.0 \
  --lookahead-collision-penalty 0.0 \
  --lookahead-distance-types qrl,euclidean

"""