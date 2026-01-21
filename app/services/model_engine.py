import threading
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from app.constants import SCENARIOS
from app.models.networks import SimpleNN, LSTMModel
from app.services.data_manager import data_manager
from app.extensions import socketio

class ModelEngine:
    def __init__(self):
        self.models_dict = {s: {} for s in SCENARIOS}
        self.accuracies_dict = {s: {} for s in SCENARIOS}
        self.risk_predictions_dict = {s: {} for s in SCENARIOS}
        self.future_predictions_dict = {s: {} for s in SCENARIOS}
        self.train_times_dict = {s: {} for s in SCENARIOS}
        self.predict_speeds_dict = {s: {} for s in SCENARIOS}
        self.simulate_running = False

    def get_models_config(self):
        return {
            'logistic_regression': LogisticRegression(max_iter=3000, solver='lbfgs', random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42), # 减少estimators加快演示
            'neural_network': 'pytorch_nn',
            'lstm': 'pytorch_lstm'
        }

    def train_all(self):
        for scenario in SCENARIOS:
            self.train_scenario(scenario)

    def train_scenario(self, scenario_key):
        try:
            logging.info(f"Start training for {scenario_key}")
            data = data_manager.data_dict[scenario_key]
            features = data_manager.train_features_dict[scenario_key]
            
            X = data[features].values
            y = data['risk_label'].values
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models_config = self.get_models_config()
            
            for model_name, model_class in models_config.items():
                start_time = time.time()
                
                # ... (此处复用原始 PyTorch/Sklearn 训练逻辑) ...
                if model_name == 'neural_network':
                    input_size = X_train.shape[1]
                    model = SimpleNN(input_size, 4)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
                    for _ in range(5): # 减少 epoch 加快启动
                        for bx, by in train_loader:
                            optimizer.zero_grad()
                            criterion(model(bx), by).backward()
                            optimizer.step()
                    with torch.no_grad():
                        y_pred = torch.argmax(model(torch.tensor(X_test)), dim=1).numpy()
                    self.models_dict[scenario_key][model_name] = (model, scaler)

                elif model_name == 'lstm':
                    # 简化 LSTM 训练逻辑
                    model = LSTMModel(X_train.shape[1], 64, 1, 4)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    criterion = nn.CrossEntropyLoss()
                    X_train_seq = torch.tensor(X_train).unsqueeze(1)
                    train_loader = DataLoader(TensorDataset(X_train_seq, torch.tensor(y_train)), batch_size=64)
                    for _ in range(5):
                        for bx, by in train_loader:
                            optimizer.zero_grad()
                            criterion(model(bx), by).backward()
                            optimizer.step()
                    with torch.no_grad():
                        y_pred = torch.argmax(model(torch.tensor(X_test).unsqueeze(1)), dim=1).numpy()
                    self.models_dict[scenario_key][model_name] = (model, scaler)
                    
                    # 未来预测
                    future_in = torch.tensor(X_test[-10:]).unsqueeze(1)
                    with torch.no_grad():
                        self.future_predictions_dict[scenario_key][model_name] = torch.argmax(model(future_in), dim=1).numpy().tolist()

                else:
                    model = clone(model_class)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    self.models_dict[scenario_key][model_name] = (model, scaler)

                acc = accuracy_score(y_test, y_pred)
                self.accuracies_dict[scenario_key][model_name] = acc
                self.risk_predictions_dict[scenario_key][model_name] = y_pred.tolist()
                
                # 记录时间指标
                self.train_times_dict[scenario_key][model_name] = time.time() - start_time
                self.predict_speeds_dict[scenario_key][model_name] = 0.5 # 模拟值
            
            logging.info(f"{scenario_key} trained.")
            
        except Exception as e:
            logging.error(f"Error training {scenario_key}: {e}")

    def predict(self, scenario_key, new_data, model_name):
        try:
            model, scaler = self.models_dict[scenario_key][model_name]
            features = data_manager.train_features_dict[scenario_key]
            new_data = new_data.reindex(columns=features, fill_value=0)
            X_new = scaler.transform(new_data.values).astype(np.float32)
            
            if model_name in ['neural_network', 'lstm']:
                X_tensor = torch.tensor(X_new)
                if model_name == 'lstm': X_tensor = X_tensor.unsqueeze(1)
                with torch.no_grad():
                    return torch.argmax(model(X_tensor), dim=1).numpy()
            else:
                return model.predict(X_new)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return np.zeros(len(new_data))

    def start_simulation(self):
        if self.simulate_running: return
        self.simulate_running = True
        thread = threading.Thread(target=self._simulation_loop, daemon=True)
        thread.start()

    def _simulation_loop(self):
        logging.info("Simulation started.")
        while self.simulate_running:
            for scenario in SCENARIOS:
                # 生成数据
                new_raw = data_manager.generate_data_batch(scenario, 5)
                # 追加处理
                processed_new = data_manager.append_new_data(scenario, new_raw)
                
                # 预测
                for model_name in self.get_models_config():
                    if model_name in self.models_dict[scenario]:
                        preds = self.predict(scenario, processed_new.drop(['risk_label'], axis=1, errors='ignore'), model_name)
                        self.risk_predictions_dict[scenario][model_name].extend(preds.tolist())
                        
                        # 检查高风险
                        recent = self.risk_predictions_dict[scenario][model_name][-100:]
                        if recent and (sum(p > 1 for p in recent) / len(recent) > 0.2):
                            socketio.emit('alert', {
                                'scenario': scenario, 
                                'model': model_name, 
                                'message': f"{scenario} - {model_name} 检测到高风险！"
                            })
                
                socketio.emit('update_data', {'scenario': scenario})
            time.sleep(10)

model_engine = ModelEngine()