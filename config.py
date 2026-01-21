import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret-key-bank-guard-ai'
    DEBUG = True
    LOG_FILE = 'app.log'