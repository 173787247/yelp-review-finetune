# YelpReviewFull 完整数据集微调训练

## 作业目标
使用完整的 YelpReviewFull 数据集训练，对比看 Acc 最高能到多少。

## 数据集信息
- **训练集**：650,000 条评论
- **测试集**：50,000 条评论
- **类别数**：5 类（1-5星评分）

## 实验策略
1. 使用完整数据集（不使用抽样）
2. 尝试不同的超参数组合：
   - 配置1：基础配置（学习率 2e-5，3 epochs）
   - 配置2：更高学习率（学习率 3e-5，3 epochs）
   - 配置3：更多训练轮数（学习率 2e-5，5 epochs）
   - 配置4：更大批次（学习率 2e-5，batch_size=32，3 epochs）
3. 对比不同配置的准确率
4. 找到最高准确率配置

## 文件说明
- `yelp_full_training.ipynb` - 主 Notebook，包含完整的训练和评估代码
- `execute_training.py` - 自动化训练脚本（用于后台运行）
- `check_completion.py` - 检查训练完成状态和文件完整性
- `results_summary.csv` - 训练结果汇总（训练完成后生成）
- `accuracy_comparison.png` - 准确率对比图表（训练完成后生成）

## 运行要求
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- GPU 推荐（训练完整数据集需要较长时间）

## 安装依赖
```bash
pip install transformers datasets accelerate evaluate torch
```

## 使用方法
1. 打开 `yelp_full_training.ipynb`
2. 按顺序运行所有 cell
3. 等待训练完成（可能需要数小时，取决于 GPU）
4. 查看结果对比和可视化图表

## 结果

### 训练结果汇总

| 配置 | 学习率 | 训练轮数 | 批次大小 | **准确率** | 训练时间 |
|------|--------|----------|----------|------------|----------|
| 配置1-基础 | 2e-05 | 3 | 16 | **70.134%** | 4.74小时 |
| 配置2-高学习率 | 3e-05 | 3 | 16 | **70.096%** | 3.76小时 |
| 配置3-更多轮数 | 2e-05 | 5 | 16 | **69.826%** | 6.17小时 |
| 配置4-大批次 | 2e-05 | 3 | 32 | **70.284%** | 3.48小时 |

### 最高准确率
**配置4（大批次）**: **70.284%**

训练完成后会生成：
- 各配置的准确率对比表格（`results_summary.csv`）
- 可视化图表（`accuracy_comparison.png`）
- 最高准确率配置信息
- 4个配置的模型文件（保存在 `models/` 目录）

## 检查训练状态
运行以下命令检查训练是否完成：
```bash
docker exec yelp-review-finetune-jupyter python /app/check_completion.py
```

或者在Windows上运行：
```bash
auto_check_completion.bat
```

## 性能优化
- 使用 FP16 混合精度训练加速
- 多进程数据加载（4 workers）
- GPU 内存固定（pin_memory）
- 当前 GPU 利用率：~92%

