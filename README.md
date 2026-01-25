# Taichi Dataset: MPM模拟与Blender渲染

本项目展示了使用Taichi进行材料点法（MPM）模拟，然后在Blender中渲染结果。它包括各种具有不同材料和物体的模拟场景，提供从物理模拟到可视化的完整管道。

## 项目结构

```
taichi_dataset/
├── simulate_script/          # MPM模拟脚本
├── particles_output/         # 模拟输出数据（NPY/PLY文件）
├── render_script/           # Blender渲染脚本
├── render_output/           # 渲染图像/视频
├── blender/                 # Blender项目文件（.blend）
├── meshes/                  # 3D网格文件（.ply）
├── run_multiple_simulations.py  # 批量模拟运行器
└── README.md
```

## 模拟场景

### 1. 原型测试 (simulate_protocol.py)
- **描述**：鸭子模型掉落到刚体地面上的原型测试
- **物体**：鸭子网格采样成粒子
- **地面**：刚体碰撞体
- **输出**：`particles_output/output_protocol/`

### 2. 多材质 (simulate_multi_material.py)
- **描述**：立方体物体掉落到沙地地面
- **物体**：立方体模型
- **地面**：容器中的沙材质
- **输出**：`particles_output/output_multi_material/`

### 3. 多米诺 (simulate_domino.py)
- **描述**：多个多米诺骨牌的连锁反应模拟
- **物体**：多个多米诺块
- **地面**：粘性表面以允许坠落/旋转
- **输出**：`particles_output/output_domino/`

### 4. 冰壶 (simulate_curling.py)
- **描述**：冰壶石模拟
- **物体**：球形冰壶石
- **地面**：光滑表面（类似冰面）
- **输出**：`particles_output/output_curling/`

### 5. 台球 MPM (simulate_billiard_mpm.py)
- **描述**：使用MPM的台球模拟
- **物体**：多个台球
- **地面**：光滑表面
- **输出**：`particles_output/output_billiard_mpm/`

### 6. 台球 非MPM (simulate_billiard_n-mpm.py)
- **描述**：使用刚体动力学的台球模拟（非MPM）
- **物体**：多个具有碰撞检测的台球
- **地面**：基于物理的地面
- **输出**：`particles_output/output_billiard_n-mpm/`

### 7. 坚硬地面 (simulate_solid_ground.py)
- **描述**：鸭子模型掉落到坚硬地面，保持网格完整性
- **物体**：鸭子网格顶点作为粒子，重构为网格
- **地面**：刚体碰撞体
- **输出**：`particles_output/output_solid_ground/`

## 材质

模拟支持各种材质类型：
- `elasticity`：弹性可变形材料
- `newtonian`：牛顿流体
- `non_newtonian`：非牛顿流体
- `plasticine`：橡皮泥类材料
- `sand`：颗粒沙材质
- `toothpaste_custom`：自定义牙膏材质

## 运行模拟

### 单次模拟
```bash
cd simulate_script
python simulate_[scenario].py -o ../particles_output/output_[scenario] -m [material]
```

示例：
```bash
python simulate_solid_ground.py -o ../particles_output/output_solid_ground -m elasticity
```

### 批量模拟
使用不同材质运行多个模拟：
```bash
python run_multiple_simulations.py
```

## 使用Blender渲染

每个模拟场景都有对应的Blender渲染脚本：

- `render_blender_[scenario].py` - 每个场景的特定渲染脚本
- `render_blender.py` - 粒子到网格转换的通用渲染脚本

### 渲染流程
1. 从`blender/`目录加载对应的`.blend`文件
2. 在Blender的Python控制台或作为脚本运行渲染脚本
3. 将渲染帧输出到`render_output/[scenario]/`

### 渲染命令示例
```bash
# 在Blender Python控制台或脚本中
import sys
sys.path.append('/path/to/taichi_dataset/render_script')
import render_blender_solid_ground
# 配置路径并运行
```

## 网格文件

- `meshes/billiard.ply`：台球网格
- `meshes/toyduck.ply`：玩具鸭网格

## 依赖项

- **Taichi**：用于MPM模拟
- **NumPy**：数值计算
- **Trimesh**：网格处理
- **Blender**：用于渲染（带Python API）

## 安装

1. 激活conda环境（gic相关配置）：
```bash
conda activate gic  # 激活gic相关的conda环境
```

2. 安装Taichi：
```bash
pip install taichi
```

3. 安装其他依赖项：
```bash
pip install numpy trimesh
```

4. 安装Blender（推荐版本4.5）

## 输出格式

- **模拟输出**：每帧包含粒子位置的NPY文件，加上metadata.json
- **渲染输出**：来自Blender的PNG/JPG图像或视频文件

## 配置

每个模拟脚本接受命令行参数：
- `-o, --output_dir`：模拟数据的输出目录
- `-m, --material`：材质类型（见材质部分）

渲染脚本具有可配置的参数，如粒子半径、分辨率、材质等。

## 注意事项

- 模拟针对CUDA GPU加速进行了优化
- 某些场景可能需要调整粒子数量以提高性能
- Blender渲染可能计算密集；根据需要调整采样和分辨率
- 该项目展示了从物理模拟到逼真渲染的完整管道
