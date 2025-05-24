from flask import Blueprint, request, jsonify
from models.auth import AccessKey
from datetime import datetime
from app import db
import uuid

access_key_bp = Blueprint('auth', __name__)

@access_key_bp.route("/generate-access-key", methods=["POST"])

def generate_access_key():
    key = str(uuid.uuid4().hex)
    try:
      new_key = AccessKey(device_id = '', access_key = key, status = False, created_at = datetime.now(), updated_at = datetime.now())
      db.session.add(new_key)
      db.session.commit()
      return jsonify({"access_key": key})
    except Exception as e:
      return jsonify({'error': e}), 404

# Authorize device with key
@access_key_bp.route("/authorize-device", methods=["POST"])

def authorize_device():
    data = request.get_json()
    access_key = data.get("access_key")
    device_id = data.get("device_id")

    if not access_key or not device_id:
        return jsonify({"detail": "Missing access_key or device_id"}), 400

    token = AccessKey.query.filter_by(access_key=access_key).first()

    if not token:
        return jsonify({"detail": "Access denied"}), 403

    if token.status == True:
        return jsonify({"detail": "Invalid or already used access key"}), 401

    # Mark key as used and store device
    token.status = True
    token.device_id = device_id
    db.session.commit()

    return jsonify({"status": "authorized"})


# Check device access
@access_key_bp.route("/check-access", methods=["GET"])

def check_access():
    device_id = request.args.get("device_id")
    if not device_id:
        return jsonify({"detail": "Missing device_id"}), 400

    token = AccessKey.query.filter_by(device_id=device_id).first()
    
    if not token:
        return jsonify({"detail": "Access denied"}), 403

    return jsonify({"status": "granted"})