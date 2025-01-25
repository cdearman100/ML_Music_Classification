from flask import Flask
from app.config import config
from app.routes import song_routes, playlist_routes, audio_routes

def create_app(config_name="default"):
    app = Flask(__name__)
    # Load configuration from the `config` dictionary
    app.config.from_object(config[config_name])

    # Register blueprints
    app.register_blueprint(song_routes.bp)
    app.register_blueprint(playlist_routes.bp)
    app.register_blueprint(audio_routes.bp)

    return app
