from app.routes.song_routes import bp as song_routes_bp
from app.routes.playlist_routes import bp as playlist_routes_bp
from app.routes.audio_routes import bp as audio_routes_bp

# Expose blueprints for app initialization
__all__ = ['song_routes_bp', 'playlist_routes_bp', 'audio_routes_bp']
