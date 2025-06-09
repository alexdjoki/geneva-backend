from app import create_app, db
from flask_cors import CORS

app = create_app()
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)