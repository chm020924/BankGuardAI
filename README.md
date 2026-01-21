# BankGuardAI 🏦 - 智能银行风控监测系统

**BankGuardAI** 是一个基于 Python 的全栈式金融风控仿真与监测平台。它集成了实时数据流模拟、多种机器学习/深度学习模型（LSTM, Neural Networks, Random Forest 等）以及交互式前端仪表盘。

该系统旨在模拟银行内部真实的业务场景，实时检测信用风险、欺诈交易、洗钱行为及网络安全威胁，并通过 WebSocket 实现毫秒级的前端预警推送。

---

## ✨ 核心功能

* **⚡ 实时仿真引擎**：后台多线程模拟生成高频交易与业务数据，模拟真实银行业务流。
* **🧠 多模型竞技场**：集成 Logistic Regression, SVM, Random Forest, GBDT, MLP (PyTorch), LSTM (PyTorch) 六种模型，实时对比在不同场景下的准确率与预测速度。
* **📊 全景可视化**：
* **风险雷达**：多维度展示风险模型评分。
* **热力图与流图**：直观展示风险聚集区域和实时变化趋势。
* **未来预测**：利用 LSTM 对未来风险趋势进行时序预测。


* **🚨 智能预警**：当检测到高风险比率超过阈值（20%）时，自动触发 WebSocket 警报并推送到前端。
* **📑 报告与导出**：支持一键生成 PDF 风险评估报告及 CSV 原始数据导出。

---

## 🏗️ 项目架构

本项目采用模块化 **MVC (Model-View-Controller)** 架构设计，结构清晰，易于扩展。

```text
BankGuardAI/
├── app/
│   ├── __init__.py          # Flask 应用工厂，初始化 SocketIO 与 LoginManager
│   ├── extensions.py        # 扩展实例 (db, socketio, login)
│   ├── constants.py         # 核心业务配置（枚举类、场景定义、风控规则）
│   ├── models/              # 深度学习模型定义
│   │   ├── __init__.py
│   │   └── networks.py      # PyTorch 模型 (SimpleNN, LSTMModel)
│   ├── services/            # 核心业务逻辑层
│   │   ├── __init__.py
│   │   ├── data_manager.py  # 数据工厂：负责数据生成、清洗、OneHot编码、状态管理
│   │   └── model_engine.py  # 智能引擎：负责模型训练、预测推理、后台仿真线程
│   ├── routes/              # 路由控制器
│   │   ├── __init__.py
│   │   ├── auth.py          # 用户认证 (Login/Logout)
│   │   ├── dashboard.py     # 页面渲染、报告生成、文件导出
│   │   └── api.py           # AJAX/Fetch 数据接口
│   ├── templates/           # 前端视图
│   │   ├── index.html       # 核心监控仪表盘 (Chart.js + Socket.io)
│   │   └── login.html       # 登录页面
│   └── static/              # 静态资源 (CSS/JS)
├── config.py                # 全局配置文件
├── run.py                   # 项目启动入口
├── requirements.txt         # 依赖包列表
└── README.md                # 项目说明文档

```

---

## 🛡️ 风控场景覆盖

系统内置 **9 大核心金融风控场景**，涵盖信用、市场、操作及合规风险：

| 场景代码 | 场景名称 | 主要风险类型 | 关键特征 (Features) | 风控手段 |
| --- | --- | --- | --- | --- |
| `enterprise_loan` | **企业贷款审批** | 信用风险 | 行业类型, 负债率, 信用分 | 财务分析, 抵押担保 |
| `credit_card` | **信用卡发放与使用** | 欺诈/信用风险 | 交易类型, 频次, 设备类型 | 交易限额, 反欺诈模型 |
| `forex_trading` | **外汇交易** | 市场风险 | 货币对, 波动率, 交易量 | 限额管理, 实时对冲 |
| `wealth_product` | **理财产品销售** | 合规/声誉风险 | 产品类型, 投资者年龄 | 适当性管理 (KYC) |
| `atm_netbank` | **ATM及网银运行** | 操作/网络风险 | 访问方式, 登录尝试次数 | 双重认证, 防火墙 |
| `interbank_derivatives` | **同业拆借与衍生品** | 流动性/市场风险 | 衍生品类型, 利率, 期限 | VAR监测, 流动性管理 |
| `aml_kyc` | **反洗钱监控 (AML)** | 法律/合规风险 | 汇款国别, PEP状态, 金额 | 异常交易监测, 黑名单 |
| `insurance_claims` | **保险理赔处理** | 欺诈风险 | 理赔类型, 过往理赔数 | 自动理赔审核 |
| `cyber_security` | **网络安全监测** | 网络风险 | 威胁类型, 影响系统数 | 入侵检测 (IDS) |

---

## 🛠️ 技术栈

### 后端 (Backend) & AI

* **Web 框架**: Flask (Python)
* **实时通信**: Flask-SocketIO (基于 Eventlet/Gevent)
* **身份验证**: Flask-Login
* **深度学习**: PyTorch (构建 LSTM 时序模型与全连接神经网络)
* **机器学习**: Scikit-learn (构建 Logistic Regression, RF, SVM, GBDT)
* **数据处理**: Pandas, NumPy
* **数据模拟**: Faker (生成逼真的用户PII及业务数据)
* **报表生成**: ReportLab (PDF 生成)

### 前端 (Frontend)

* **UI 框架**: Bootstrap 5 (响应式布局, Dark Mode 支持)
* **可视化**: Chart.js 4.x (Bar, Radar, Line, Doughnut)
* **高级图表**: Chartjs-chart-matrix (热力图), Chartjs-plugin-zoom (缩放交互)
* **交互逻辑**: WebSocket 客户端, Sortable.js (拖拽排序)

---

## 🚀 快速开始

### 1. 环境准备

请确保您的环境已安装 Python 3.8 或更高版本。

```bash
# 克隆项目 (假设您已下载代码)
cd BankGuardAI

# 创建虚拟环境 (推荐)
python -m venv venv

# 激活环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 2. 安装依赖

```bash
pip install -r requirements.txt

```

### 3. 运行系统

```bash
python run.py

```

> **注意**: 首次启动时，系统会自动生成模拟数据并训练所有场景下的机器学习模型。这可能需要 **1-3 分钟**，请耐心等待控制台出现 `Running on http://127.0.0.1:5000` 提示。

### 4. 登录系统

打开浏览器访问 `http://127.0.0.1:5000`。

* **管理员账号** (可查看所有场景):
* 用户名: `admin`
* 密码: `admin`


* **普通用户账号** (仅查看部分场景):
* 用户名: `user`
* 密码: `user`



---

## 🖥️ 核心类与方法说明

### `DataManager` (services/data_manager.py)

单例模式管理全局数据。

* `initialize_data()`: 启动时加载或生成 CSV 数据，并进行 OneHot 编码预处理。
* `generate_data_batch()`: 根据不同场景的统计分布（如正态分布、泊松分布）生成模拟数据。

### `ModelEngine` (services/model_engine.py)

模型生命周期管理。

* `train_all()`: 遍历所有场景，对数据进行 `StandardScaler` 标准化后，训练 LR, RF, SVM, NN, LSTM 等模型。
* `_simulation_loop()`: 守护线程，每 10 秒生成一批新数据，调用模型预测，并通过 SocketIO 推送更新。

### `LSTMModel` (models/networks.py)

专用于时序风险预测。

* 输入：历史 N 个时间步的特征向量。
* 输出：未来风险等级的概率分布。

---

## 📜 许可证

本项目仅供学习与研究使用。

---

*Made with ❤️ by BankGuardAI Team*