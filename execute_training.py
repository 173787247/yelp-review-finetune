#!/usr/bin/env python
"""
执行 YelpReviewFull 完整数据集训练
这个脚本会执行 notebook 中的训练代码
"""
import sys
import os
import time
from datetime import datetime

print("=" * 80)
print("YelpReviewFull 完整数据集训练")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 导入必要的库
try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    import torch
    import numpy as np
    import evaluate
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ 所有依赖库已导入")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("正在安装依赖...")
    os.system("pip install -q transformers datasets accelerate evaluate matplotlib pandas")
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    import torch
    import numpy as np
    import evaluate
    import pandas as pd
    import matplotlib.pyplot as plt

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 1. 加载数据集
print("\n" + "=" * 80)
print("步骤 1: 加载数据集")
print("=" * 80)
print("正在加载 YelpReviewFull 数据集...")
dataset = load_dataset("yelp_review_full")
print(f"✓ 数据集加载完成！")
print(f"  训练集大小: {len(dataset['train']):,}")
print(f"  测试集大小: {len(dataset['test']):,}")

# 2. 数据预处理
print("\n" + "=" * 80)
print("步骤 2: 数据预处理")
print("=" * 80)
model_name = "bert-base-uncased"
print(f"正在加载分词器: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("开始对数据集进行分词处理...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)
print("✓ 分词处理完成！")

# 准备训练和测试数据集
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

print(f"✓ 训练集准备完成: {len(train_dataset):,} 条")
print(f"✓ 测试集准备完成: {len(test_dataset):,} 条")

# 3. 定义评估指标
print("\n" + "=" * 80)
print("步骤 3: 定义评估指标")
print("=" * 80)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print("✓ 评估指标定义完成")

# 4. 训练配置
configs = [
    {
        "name": "Config 1 (Base)",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
    },
    {
        "name": "Config 2 (High LR)",
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
    },
    {
        "name": "Config 3 (More Epochs)",
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
    },
    {
        "name": "Config 4 (Large Batch)",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "weight_decay": 0.01,
    }
]

all_results = []

# 5. 训练每个配置
for idx, config in enumerate(configs, 1):
    print("\n" + "=" * 80)
    print(f"配置 {idx}/4: {config['name']}")
    print("=" * 80)
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_train_epochs']}")
    print(f"批次大小: {config['per_device_train_batch_size']}")
    
    start_time = time.time()
    
    # 加载模型
    print("正在加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=5
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=f"./results/{config['name'].replace(' ', '_')}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,  # 加速数据加载
        dataloader_pin_memory=True,  # 使用固定内存加速
        report_to=None,  # 禁用wandb等外部报告
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 评估
    print("开始评估...")
    results = trainer.evaluate()
    accuracy = results['eval_accuracy']
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print(f"\n{config['name']} 结果:")
    print(f"  准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  训练时间: {hours}小时 {minutes}分钟")
    
    # 保存模型
    trainer.save_model(f"./models/{config['name'].replace(' ', '_')}")
    print(f"  模型已保存到: ./models/{config['name'].replace(' ', '_')}")
    
    all_results.append({
        "配置": config["name"],
        "学习率": config["learning_rate"],
        "训练轮数": config["num_train_epochs"],
        "批次大小": config["per_device_train_batch_size"],
        "准确率": accuracy,
        "训练时间(小时)": elapsed_time / 3600
    })

# 6. 结果汇总
print("\n" + "=" * 80)
print("所有配置的结果对比")
print("=" * 80)
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# 找到最高准确率
best_result = results_df.loc[results_df['准确率'].idxmax()]
print("\n" + "=" * 80)
print("最高准确率配置:")
print("=" * 80)
print(f"配置: {best_result['配置']}")
print(f"准确率: {best_result['准确率']:.4f} ({best_result['准确率']*100:.2f}%)")
print(f"学习率: {best_result['学习率']}")
print(f"训练轮数: {best_result['训练轮数']}")
print(f"批次大小: {best_result['批次大小']}")

# 保存结果
results_df.to_csv("results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: results_summary.csv")

# 可视化
print("\n正在生成可视化图表...")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(results_df['配置'], results_df['准确率'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Accuracy Comparison Across Configurations', fontsize=14, fontweight='bold')
plt.xlabel('Configuration', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([0, 1])
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(results_df['准确率']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(results_df['配置'], results_df['准确率'] * 100, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Accuracy (%) Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Configuration', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(results_df['准确率'] * 100):
    plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为: accuracy_comparison.png")

print("\n" + "=" * 80)
print(f"训练完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

