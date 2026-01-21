from flask import Blueprint, jsonify, login_required
from app.constants import SCENARIOS
from app.services.data_manager import data_manager
from app.services.model_engine import model_engine
import numpy as np

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/data/<scenario>', methods=['GET'])
@login_required
def get_data(scenario):
    if scenario in SCENARIOS:
        models = model_engine.get_models_config().keys()
        recent_preds = {m: model_engine.risk_predictions_dict[scenario].get(m, [])[-100:] for m in models}
        
        return jsonify({
            'recent_data': data_manager.data_dict[scenario].tail(10).to_dict(orient='records'),
            'risk_distributions': {m: recent_preds[m] for m in models},
            'risk_levels': {m: np.bincount(recent_preds[m], minlength=4).tolist() for m in models},
            'accuracies': model_engine.accuracies_dict[scenario],
            'future_predictions': model_engine.future_predictions_dict.get(scenario, {})
        })
    return jsonify({'error': 'Invalid scenario'}), 400