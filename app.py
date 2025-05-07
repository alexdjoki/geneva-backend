from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from settings import Config

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    from controllers.openai import openai_bp
    app.register_blueprint(openai_bp, url_prefix='/openai')

    from controllers.chat_history import chat_history_bp
    app.register_blueprint(chat_history_bp, url_prefix='/chat-history')

    from controllers.product_history import product_history_bp
    app.register_blueprint(product_history_bp, url_prefix='/product-history')
    return app
