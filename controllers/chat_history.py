from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from app import db
from datetime import datetime

chat_history_bp = Blueprint('chat-history', __name__)

@chat_history_bp.route('/list', methods=['POST'])

def list():
  data = request.get_json()
  user_id = data.get('user_id')
  try:
    chathistory = ChatHistory.query.all()
    list = [item.to_dict() for item in chathistory]
    return jsonify(list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@chat_history_bp.route('/delete', methods=['POST'])

def delete():
  data = request.get_json()
  user_id = data.get('user_id')
  chat_id = data.get('chat_id')
  try:
    chathistory = ChatHistory.query.filter_by(user_id = user_id, chat_id = chat_id).delete()
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500