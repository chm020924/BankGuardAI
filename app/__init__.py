from flask import Flask
from .extensions import socketio, login_manager
from .services.data_manager import data_manager
from .services.model_engine import model_engine
import logging

def create_app(config_class='config.Config'):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Extensions
    socketio.init_app(app)
    login_manager.init_app(app)

    # Logging
    logging.basicConfig(filename=app.config.get('LOG_FILE', 'app.log'), 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Register Blueprints
    from .routes.auth import auth_bp
    from .routes.dashboard import dashboard_bp
    from .routes.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)

    # Initialization Logic (Data & Models)
    with app.app_context():
        data_manager.initialize_data()
        model_engine.train_all()
        model_engine.start_simulation()

    return app

@socketio.on('connect')
def handle_connect():
    from flask_socketio import emit
    emit('update_data', {'message': 'Connected to BankGuardAI'})