from app import create_app
import os

app = create_app(config_name=os.getenv("FLASK_ENV", "default"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment, default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to 0.0.0.0 for public access
