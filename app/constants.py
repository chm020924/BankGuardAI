from enum import Enum

class RiskType(Enum):
    CREDIT = '信用风险'
    OPERATION = '操作风险'
    FRAUD = '欺诈风险'
    MARKET = '市场风险'
    COMPLIANCE = '合规风险'
    REPUTATION = '声誉风险'
    LIQUIDITY = '流动性风险'
    LEGAL = '法律风险'
    NETWORK = '网络风险'

# 场景配置
SCENARIOS = {
    'enterprise_loan': {
        'name': '企业贷款审批',
        'risks': [RiskType.CREDIT],
        'controls': '尽职调查、财务分析、信用评级、抵押担保',
        'categorical_features': ['industry_type'],
        'categories': [['tech', 'finance', 'retail', 'manufacturing']],
    },
    'credit_card': {
        'name': '信用卡发放与使用',
        'risks': [RiskType.CREDIT, RiskType.OPERATION, RiskType.FRAUD],
        'controls': '风控模型评分、交易限额、反欺诈系统',
        'categorical_features': ['transaction_type', 'device_type'],
        'categories': [['purchase', 'cash_advance', 'balance_transfer'], ['mobile', 'desktop', 'tablet']],
    },
    'forex_trading': {
        'name': '外汇交易',
        'risks': [RiskType.MARKET],
        'controls': '限额管理、实时汇率监控、对冲策略',
        'categorical_features': ['currency_pair'],
        'categories': [['USD/EUR', 'USD/JPY', 'GBP/USD']],
    },
    'wealth_product': {
        'name': '理财产品销售',
        'risks': [RiskType.COMPLIANCE, RiskType.REPUTATION],
        'controls': '投资者适当性管理、信息披露制度',
        'categorical_features': ['product_type'],
        'categories': [['stock', 'bond', 'fund', 'crypto']],
    },
    'atm_netbank': {
        'name': 'ATM及网银系统运行',
        'risks': [RiskType.OPERATION, RiskType.NETWORK],
        'controls': '系统加密、防火墙、双重认证、应急备份',
        'categorical_features': ['access_method', 'device_type'],
        'categories': [['ATM', 'Netbank', 'MobileApp'], ['mobile', 'desktop', 'ATM_machine']],
    },
    'interbank_derivatives': {
        'name': '同业拆借与衍生品交易',
        'risks': [RiskType.MARKET, RiskType.LIQUIDITY],
        'controls': 'VAR监测、限额控制、流动性管理',
        'categorical_features': ['derivative_type'],
        'categories': [['swap', 'option', 'future']],
    },
    'aml_kyc': {
        'name': '反洗钱监控（AML/KYC）',
        'risks': [RiskType.LEGAL, RiskType.COMPLIANCE],
        'controls': '客户身份验证、异常交易监测、报告机制',
        'categorical_features': ['source_country', 'destination_country'],
        'categories': [['USA', 'China', 'Japan', 'Germany', 'UK'], ['USA', 'China', 'Japan', 'Germany', 'UK']],
    },
    'insurance_claims': {
        'name': '保险理赔处理',
        'risks': [RiskType.FRAUD, RiskType.OPERATION],
        'controls': '理赔审核、反欺诈检测、数据验证',
        'categorical_features': ['claim_type'],
        'categories': [['health', 'auto', 'property', 'life']],
    },
    'cyber_security': {
        'name': '网络安全监测',
        'risks': [RiskType.NETWORK, RiskType.FRAUD],
        'controls': '入侵检测、防火墙配置、事件响应',
        'categorical_features': ['threat_type'],
        'categories': [['phishing', 'malware', 'ddos', 'insider']],
    }
}