from flask import Blueprint, render_template, send_file, current_user, login_required
from app.constants import SCENARIOS
from app.services.data_manager import data_manager
from app.services.model_engine import model_engine
import numpy as np
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    allowed_scenarios = list(SCENARIOS.keys()) if current_user.role == 'admin' else list(SCENARIOS.keys())[:3]
    stats = {}
    
    for scenario in allowed_scenarios:
        data = data_manager.data_dict.get(scenario, [])
        if len(data) == 0: continue
        
        recent_preds = {m: model_engine.risk_predictions_dict[scenario].get(m, [])[-100:] for m in model_engine.get_models_config()}
        alerts = {m: (sum(p > 1 for p in pr) / len(pr) > 0.2) if pr else False for m, pr in recent_preds.items()}
        
        stats[scenario] = {
            'name': SCENARIOS[scenario]['name'],
            'total_transactions': len(data),
            'risk_count': int((data['risk_label'] > 1).sum()),
            'controls': SCENARIOS[scenario]['controls'],
            'accuracies': model_engine.accuracies_dict[scenario],
            'alerts': alerts,
            'train_times': model_engine.train_times_dict.get(scenario, {}),
            'predict_speeds': model_engine.predict_speeds_dict.get(scenario, {})
        }
    
    return render_template('index.html', stats=stats, scenarios_list=allowed_scenarios, models_list=list(model_engine.get_models_config().keys()))

@dashboard_bp.route('/report/<scenario>')
@login_required
def generate_report(scenario):
    if scenario not in SCENARIOS: return "Invalid Scenario", 400
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, f"{SCENARIOS[scenario]['name']} Risk Report")
    c.drawString(100, 730, f"Total Events: {len(data_manager.data_dict[scenario])}")
    # ... 更多PDF内容 ...
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"{scenario}_report.pdf", mimetype='application/pdf')

@dashboard_bp.route('/export/<scenario>')
@login_required
def export_data(scenario):
    if scenario == 'all':
        # 简化处理，仅示例
        return "Export All Not Implemented in Demo", 200
    if scenario in SCENARIOS:
        buffer = io.StringIO()
        data_manager.data_dict[scenario].to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(io.BytesIO(buffer.getvalue().encode()), as_attachment=True, download_name=f"{scenario}_data.csv", mimetype='text/csv')
    return "Invalid", 400