# Deep Learning Hyperparameters

## 処理の流れ: データ準備→ モデル構築→ 最適化設定→ 学習スケジュール→ 正規化

## Preprocessing: データの前処理
# データがどう扱われるかが決まる
- 正規化パラメータ
- Shuffle: シャッフルの有無
- Sampler: サンプラー（WeightedRandomSampler など）
- データ拡張(augmentation)の強度

## Architecture: モデル構造
# どんなモデルを使うか
- depth: 層の数
- hidden size: 隠れ次元
- Attention head
- Activation function: 活性化関数
- CNN の kernel/stride/padding
- dropout 率

## Optimization: 最適化設定
# どう学習させるか
- Learning rate: 学習率
- Optimizer (SGD, Adam, etc.): 最適化手法・アルゴリズム
- Momentum: モメンタム（勾配の慣性を持たせる）
- β1, β2 (Adam)
- Weight decay: 重み減衰
- Batch size: バッチサイズ

## Training Schedule: 学習スケジュール
# 学習率の変化や学習の進め方
- LR scheduler (cosine, step decay, etc.): 学習率スケジューラ
- Warmup steps
- Epochs: エポック数
- patience of early stopping: Early stoppingのpatience

## Regularization
# 学習の安定化や過学習対策
- Dropout: 学習側のdropout
- BatchNorm momentum
- Data augmentation strength


# 実験で使う全パラメータ

## データセット
- num_train: train データの数
- num_test: test データの数

## モデル
- num_modality1_layer: モダリティ1の層数
- num_modality2_layer: モダリティ2の層数
- num_after_layer: 結合後の層数
- activation_function: 活性化関数
- dropout: ドロップアウト率
- optimizer: 最適化手法・アルゴリズム
- batch_size: バッチサイズ



## tree -L 3
# 最終更新: 2025年12月9日
.
├── README.md
├── configs
│   ├── config.yaml
│   ├── hyperparameter.yaml
│   └── runtime.yaml
├── data
│   └── network.py
├── data_gen
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── data_utils.cpython-312.pyc
│   │   ├── dijkstra.cpython-312.pyc
│   │   ├── generate_data.cpython-312.pyc
│   │   ├── network_core.cpython-312.pyc
│   │   ├── preprocess.cpython-312.pyc
│   │   └── save_data.cpython-312.pyc
│   ├── data_utils.py
│   ├── dijkstra.py
│   ├── generate_data.py
│   ├── network.py
│   ├── network2.py
│   ├── network_core.py
│   ├── networkenv_ex_re.py
│   ├── preprocess.py
│   └── save_data.py
├── experiments
│   └── exp_20251208_140509
│       ├── config.yaml
│       └── hyperparameters.yaml
├── model
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── fully_net.cpython-312.pyc
│   ├── fully_net.py
│   └── fully_net2.py
├── multimodal_net_training.py
├── results
│   ├── checkpoints
│   │   ├── best_model.pth
│   │   ├── checkpoint_model.pth
│   │   ├── chekpoint_model.pth
│   │   └── final_model.pth
│   ├── data
│   │   ├── datasets
│   │   └── raw
│   ├── figures
│   │   ├── batch_size=128_lr=1e-2.png
│   │   ├── batch_size=128_lr=1e-3.png
│   │   ├── batch_size=128_lr=1e-4.png
│   │   ├── batch_size=128_lr=1e-5.png
│   │   ├── batch_size=128_lr=3e-3.png
│   │   └── batch_size=128_lr=5e-5.png
│   ├── logs
│   │   ├── train_log.pt
│   │   └── val_log.pt
│   ├── lr_range_test
│   │   ├── lr_range_test.png
│   │   └── lr_range_test_15.png
│   ├── outputs
│   │   └── outputs.csv
│   └── test
├── scripts
│   ├── __pycache__
│   │   └── train.cpython-312.pyc
│   ├── inference.py
│   ├── train.py
│   └── visualize.py
├── test
│   ├── tamesi.py
│   ├── test.ipynb
│   └── test.py
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   ├── config_loader.cpython-312.pyc
    │   ├── early_stopping.cpython-312.pyc
    │   ├── lr_range_test.cpython-312.pyc
    │   └── metrics.cpython-312.pyc
    ├── config_loader.py
    ├── early_stopping.py
    ├── lr_range_test.py
    └── metrics.py

24 directories, 62 files