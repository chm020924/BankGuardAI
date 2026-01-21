from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    print("🚀 BankGuardAI is starting...")
    print("📊 Initializing models and data (this may take a minute)...")
    socketio.run(app, debug=True, use_reloader=False) # use_reloader=False 防止线程重复启动