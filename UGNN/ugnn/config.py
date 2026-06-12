import os
import random
import argparse

import numpy as np
import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--dataset', '-D', type=str, default='actor')#texasChameleon
    parser.add_argument('--baseseed', '-S', type=int, default=42)
    parser.add_argument('--hidden', '-H', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--dp1', type=float, default=0.0)
    parser.add_argument('--dp2', type=float, default=0.9)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--forcing', type=int, default=0, choices=[0, 1])
    parser.add_argument('--addself', '-A', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model', '-M', type=str, default='EdgeNCSAGE',
                        choices=['EdgeNCSAGE', 'EdgeNCGCN'])
    parser.add_argument('--threshold', '-T', type=float, default=0.3)
    parser.add_argument('--finalagg', type=str, default='add')

    # 消融实验
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'wo_prior', 'wo_local_prior',
                                 'wo_consistency', 'wo_refinement',
                                 'wo_dual_uncertainty', 'wo_dual_channel'])

    # 门控超参
    parser.add_argument('--gate-temperature', type=float, default=1.0)
    parser.add_argument('--gate-min', type=float, default=0.0)
    parser.add_argument('--gate-max', type=float, default=1.0)
    parser.add_argument('--gate-hidden', type=int, default=256)
    parser.add_argument('--gate-dropout', type=float, default=0.5)
    parser.add_argument('--gate-epochs', type=int, default=20)
    parser.add_argument('--gate-batch-size', type=int, default=65536)#65536 #10248

    parser.add_argument('--gate-update-freq', type=int, default=5,
                        help='门控固定更新频率，每 N epoch 强制更新一次')

    # 贡献一：数据驱动先验
    parser.add_argument('--kl-beta', type=float, default=1e-2,
                        help='数据驱动 KL 散度权重')

    # 贡献二：一致性不确定性
    parser.add_argument('--consist-beta', type=float, default=5e-3,
                        help='一致性正则化权重')
    parser.add_argument('--mc-samples', type=int, default=10,
                        help='MC Dropout 采样次数')

    # 贡献三：结构修复
    parser.add_argument('--refine-beta', type=float, default=1e-4,
                        help='修复边质量损失权重')
    parser.add_argument('--epi-threshold', type=float, default=0.5,
                        help='认知不确定性阈值')
    parser.add_argument('--ale-threshold', type=float, default=0.7,
                        help='偶然不确定性阈值')
    parser.add_argument('--knn-k', type=int, default=3,
                        help='kNN 修复邻居数')
    parser.add_argument('--refine-freq', type=int, default=10,
                        help='结构修复执行频率')
    parser.add_argument('--max-repair-nodes', type=int, default=500,
                        help='每次修复最大节点数')

    # 辅助参数
    parser.add_argument('--uncertainty-gamma', type=float, default=1.0)
    parser.add_argument('--log-uncertainty', type=int, default=0,
                        choices=[0, 1])

    # 数据划分保存与复用
    parser.add_argument(
        '--split-dir',
        type=str,
        default='./splits',
        help='保存或加载 train/val/test masks 的目录，用于保证所有 baseline 使用同一划分'
    )
    parser.add_argument(
        '--use-saved-splits',
        type=int,
        default=1,
        choices=[0, 1],
        help='若 split 文件存在，是否优先加载已有划分'
    )
    parser.add_argument(
        '--save-splits',
        type=int,
        default=1,
        choices=[0, 1],
        help='是否保存当前生成的数据划分'
    )

    return parser.parse_args()
