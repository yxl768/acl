# ACL Knee MRI Segmentation Project
  
## 项目背景
膝关节前交叉韧带（ACL）损伤是临床常见的创伤性疾病之一。MRI 是诊断和评估 ACL 损伤的重要手段。对 ACL 做到精确的像素级分割，有助于术前规划、损伤量化、术后随访以及自动化度量指标的计算。

本项目实现了一个可复现的医学图像分割流水线（单通道 MRI 切片），目标是将前交叉韧带从切片中分割出来并输出像素级掩码。

面临的主要挑战：
- 解剖结构细长且对比度弱，边界模糊；
- 扫描协议/厂家差异导致图像分布不一致；
- 正负类样本严重不平衡（ACL 像素占比很小）。

项目目标：
- 构建从数据加载、清洗、增强、训练到评估的完整流水线；
- 提供可复制的训练脚本与可视化输出，便于撰写 Word/PDF 报告；
- 给出基线模型的性能，并提出可行的改进方案。

## 相关工作（简要）
U-Net 系列网络（U-Net、Attention U-Net、UNet++、TransUNet）是医学图像分割领域的常用架构。常用策略包括：结合 Dice/IoU 损失与交叉熵损失以缓解类不平衡、使用自适应优化器或学习率调度、引入更强的数据增强与后处理。本仓库实现了 U-Net 基线，并在代码中保留了便于扩展的模块接口。

## 目录结构（重要文件）
- `Raw Datset Knee MRI slices/images/`：原始输入单通道 MRI 图像
- `Raw Datset Knee MRI slices/masks/`：对应的二值掩码（0/1）
- `Raw Datset Knee MRI slices/train.py`：训练与评估主脚本（包含数据集、网络、训练循环、评估与可视化）
- `output/`：训练输出（模型权重、可视化图、CSV 指标等）

## 1. 项目概述
本项目针对膝关节前交叉韧带（ACL）的单通道MRI切片，完成前交叉韧带的像素级语义分割任务。该任务属于医学图像分割（Segmentation）范畴，不是分类（Classification）或目标检测（Detection）。

本项目的目标是：
- 从数据预处理、模型设计、损失函数与优化器选择、训练过程、模型评估等方面，进行深入分析
- 提交一份详细的报告，记录整个模型工作流程与结果
- 使用可视化图表展现训练曲线与分割结果，支持 Word/PDF 格式报告的编写

## 2. 任务类型选择

### 选择任务：分割（Segmentation）
- 任务类型：二分类语义分割
- 输入：单通道膝关节MRI影像
- 输出：前交叉韧带ACL的像素级掩码

### 备选任务说明
- 分类：适合整体疾病类型判断，但不适合定位韧带区域
- 检测：适合检测目标位置边界框，但无法给出精确像素级形状
- 因此本项目选择分割任务，更符合ACL定位与结构恢复的需求

## 3. 数据预处理

### 3.1 数据加载
- 数据目录组织为：
  - `Raw Datset Knee MRI slices/images`：输入MRI图像
  - `Raw Datset Knee MRI slices/masks`：对应二值掩码
- `train.py` 中 `ACLDataset` 类负责读取这两个目录，并按文件名匹配图像与掩码
- 样本只选取同时存在图像和掩码的文件名交集，保证训练对齐

### 3.2 数据清洗
- 通过 `ACLDataset` 读取目录中的文件名列表，并筛选交集文件
- 删除或忽略缺失对应掩码的图像，避免样本不一致
- 对无效文件、隐藏文件和非图像文件过滤
- 如果 `self.images` 为 0，则抛出异常提示路径或文件匹配问题

### 3.3 数据增强
- 训练阶段可选择开启 `--augment` 参数
- 采用的增强方式包括：
  - 随机水平翻转
  - 随机垂直翻转
  - 随机旋转 10 度
- 这些增强方法增加样本多样性，减少模型过拟合，并保持解剖结构特征

### 3.4 标准化与归一化
- 图像先通过 `transforms.ToTensor()` 转为 `[0, 1]`
- 然后通过 `transforms.Normalize([0.5], [0.5])` 归一化为近似 `[-1, 1]`
- 掩码使用 `transforms.Resize(..., interpolation=NEAREST)`，保证掩码像素不被连续值插值污染
- 最终掩码通过 `mask = (mask > 0.5).float()` 转为二值Tensor

### 3.5 注意点
- 处理单通道图像时，模型输入通道数为 1
- 如果使用彩色数据或多模态MRI，可改为 3 通道或更多通道

## 4. 模型设计

### 4.1 架构选择：CNN-UNet
本项目选择经典卷积神经网络架构 `StandardUNet`，理由如下：
- U-Net 是医学图像分割领域的标准架构
- 能同时保留语义信息与空间细节
- 对小样本场景稳定性强

### 4.2 模型结构
#### 4.2.1 编码器（下采样）
- `ConvBlock`：两层卷积 + BatchNorm + ReLU + Dropout
- 每个下采样层先做 `MaxPool2d(2)`，再做卷积块
- 特征通道逐层扩大：16 -> 32 -> 64 -> 128 -> 256

#### 4.2.2 解码器（上采样）
- `UpBlock`：双线性上采样 + 跳跃连接 + `ConvBlock`
- 上采样后与对应编码器层特征拼接
- 保留高分辨率空间信息，复原目标结构

#### 4.2.3 输出层
- 最后一层使用 `nn.Conv2d(base_filters, 1, kernel_size=1)`
- 输出单通道概率图，并通过 `torch.sigmoid()` 转为 [0,1]

### 4.3 CNN vs Vision Transformer
- Vision Transformer（ViT）适用于大规模数据和全局依赖建模
- 本项目数据集规模较小，输入图像简单，且任务为边界清晰的医学分割
- 因此选择轻量 CNN-UNet 优于复杂 ViT，减少过拟合风险

## 5. 优化器选择

### 5.1 Adam 优化器
- 采用 `torch.optim.Adam(model.parameters(), lr=args.lr)`
- 默认学习率为 `1e-3`

### 5.2 选择理由
- Adam 是自适应学习率优化器，能够快速收敛
- 对批量大小较小、噪声较高的训练表现稳定
- 相较于 SGD，Adam 通常能更快达到合理精度

### 5.3 备选优化器
- SGD + momentum：一般需要更细致学习率调度，适合大规模训练
- RMSprop：对稀疏梯度相对稳定

## 6. 损失函数设计

### 6.1 BCE + Dice 组合损失
- `ComboLoss` = `0.5 * BCELoss + 0.5 * DiceLoss`
- `BCELoss` 提供像素级二分类监督
- `DiceLoss` 关注前交叉韧带区域的重叠度，减轻类别不平衡影响

### 6.2 损失公式说明
- BCE：衡量像素预测概率与真实标签差异
- Dice：`1 - (2 * intersection / (union + eps))`
- 组合后既考虑像素精度，也考虑结构一致性

### 6.3 备选损失函数
- 交叉熵损失：适合多类别分割
- Focal Loss：适合极度类别不平衡
- IoU Loss：直接优化重叠指标

## 7. 训练过程

### 7.1 数据划分策略
默认使用验证集 15%、测试集 10%:
- 训练集：75%
- 验证集：15%
- 测试集：10%

划分通过 `random_split()` 完成，保证随机但可复现。

### 7.2 训练配置
主要训练参数：
- `batch_size=4`
- `epochs=5`（可根据资源延长至 20+）
- `num_workers=0`
- `patience=3`：连续 3 轮验证损失不下降则早停
- `use_amp`：GPU 可选混合精度训练

### 7.3 训练流程
1. 加载数据集并构建 `DataLoader`
2. 初始化模型、损失、优化器
3. 每个 epoch:
   - `train_epoch()` 进行训练
   - `evaluate_epoch()` 对验证集评估
   - 比较验证损失并保存最优模型
4. 训练结束后，加载最优模型进行测试集评估

### 7.4 日志与监控
- 使用 `tqdm` 显示训练与验证进度
- 保存训练历史到 `training_metrics.csv`
- 保存曲线图与预测结果图，方便后续分析

### 7.5 早停机制
- 如果验证损失连续 `patience` 轮没有改善，则停止训练
- 有助于防止模型在验证集上过拟合

## 8. 模型评估

### 8.1 评价指标
在测试集上计算以下指标：
- Dice coefficient
- IoU
- Precision
- Recall
- F1 score
- Accuracy

这些指标可以衡量分割精度、漏检率和整体一致性。

### 8.2 混淆矩阵
- 使用 `plot_confusion_matrix()` 可视化 TP / FP / FN / TN
- 这有助于判断模型在背景与ACL像素间的区分能力

### 8.3 结果可视化图表
建议在报告中插入以下图像：
- `training_history.png`：训练/验证损失曲线
- `iou_f1_curves.png`：训练/验证 IoU 与 F1 曲线
- `confusion_matrix.png`：测试集混淆矩阵
- `sample_predictions/sample_1.png`：输入、真实掩码、预测掩码、叠加结果

### 8.4 分割结果分析
- 若 Dice/IoU 不理想，可分析模型输出是否存在“边界模糊”、“漏分割”或“过分割”
- 通过 `sample_predictions` 中的结果图，可以直观判断模型对ACL形状的拟合情况

## 9. 代码模块说明

### 9.1 数据模块
- `ACLDataset`：图像与掩码匹配加载
- `get_transforms()`：构造训练/评估变换流程

### 9.2 模型模块
- `ConvBlock`：基本卷积特征提取块
- `UpBlock`：上采样与跳跃连接块
- `StandardUNet`：完整UNet模型

### 9.3 损失与指标模块
- `DiceLoss`：Dice系数损失
- `ComboLoss`：BCE+Dice混合损失
- `get_binary_metrics()`：评价指标计算

### 9.4 训练与评估模块
- `train_epoch()`：训练一个 epoch
- `evaluate_epoch()`：验证/测试一个 epoch
- `save_checkpoint()`：保存最优模型
- `save_sample_predictions()`：保存预测可视化结果

## 10. 实验结果与分析

### 10.1 结果展示位置
在 Word/PDF 报告中，可按以下顺序插入实验结果：
1. 数据预处理流程图或数据目录结构示意图
2. 模型架构图（UNet 编码器-解码器结构）
3. `training_history.png` 训练/验证损失曲线
4. `iou_f1_curves.png` IoU / F1 曲线
5. `confusion_matrix.png` 混淆矩阵
6. `sample_predictions` 中任意 2-4 张分割结果图

### 10.2 评估报告示例内容
- 训练损失随 epoch 下降，验证损失稳定
- IoU 和 F1 曲线反映模型分割质量
- 混淆矩阵展示模型对ACL像素与背景像素的区分能力
- 样本图验证模型是否能准确重建前交叉韧带形状

## 11. 总结与未来工作

### 11.1 优点
- 选择了适合医学分割任务的 UNet 架构
- 结合 BCE 与 Dice 损失，有效应对类别不平衡
- 通过训练/验证/测试集划分与早停机制，增强模型泛化能力
- 提供了完整的可视化和指标记录方案

### 11.2 缺点
- 目前仅单通道输入，未考虑多模态 MRI 或多序列信息
- 数据增强方式较基础，可扩展为更多仿射、噪声、弹性变形等
- 模型结构为基础 UNet，尚未使用更先进的 Attention 或 Transformer 模型
- 如果数据量较小，模型仍可能出现过拟合

### 11.3 改进方案
- 引入 Attention U-Net / UNet++ / TransUNet 等更强分割结构
- 增加数据增强策略，如 elastic deformation、随机亮度/对比度变化
- 使用学习率调度器（如 Cosine Annealing、ReduceLROnPlateau）
- 引入交叉验证、模型集成或迁移学习
- 若数据允许，可尝试多通道融合或多任务学习

### 11.4 未来工作方向
- 扩展到ACL损伤分级、病灶量化或术后恢复评估
- 将模型部署为临床辅助工具，结合前处理与后处理模块
- 研究更大规模膝关节MRI数据集，并比较不同模型效果

## 12. 运行说明

### 12.1 训练命令
```bash
python "Raw Datset Knee MRI slices/train.py"
```

### 12.2 自定义参数示例
```bash
python "Raw Datset Knee MRI slices/train.py" --img_dir "Raw Datset Knee MRI slices/images" --mask_dir "Raw Datset Knee MRI slices/masks" --epochs 20 --batch_size 8 --augment
```

### 12.3 输出文件说明
- `output/training_history.png`
- `output/iou_f1_curves.png`
- `output/confusion_matrix.png`
- `output/training_metrics.csv`
- `output/sample_predictions/`：预测结果图
- `output/best_model.pth`




