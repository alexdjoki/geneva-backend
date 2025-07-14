from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from models.devices import Devices
from app import db
from datetime import datetime
from sqlalchemy import func

chat_history_bp = Blueprint('chat-history', __name__)


@chat_history_bp.route('/chat-content', methods=['POST'])

def chatContent():
  data = request.get_json()
  chat_id = data.get('chat_id')
  try:
    chat_content = ChatHistory.query.filter_by(chat_id = chat_id).all();
    list = [item.to_dict() for item in chat_content]
    return jsonify(list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@chat_history_bp.route('/list', methods=['POST'])

def list():
  data = request.get_json()
  device_id = data.get('user_id')
  
  try:
    device_info = Devices.query.filter_by(device_id = device_id).first()
    chathistory = ChatHistory.query.filter_by(user_id = device_info.email).all()
    # list = [item.to_dict() for item in chathistory]
    if not chathistory or len(chathistory) == 0:
       chathistory = ChatHistory.query.order_by(func.random()).limit(30).all()
    result_list = [{'id': item.id, 'question': item.question, 'level': item.level, 'chat_id': item.chat_id} for item in chathistory]
    return jsonify(result_list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@chat_history_bp.route('/delete', methods=['POST'])

def delete():
  data = request.get_json()
  device_id = data.get('user_id')
  chat_id = data.get('chat_id')
  try:
    device_info = Devices.query.filter_by(device_id = device_id).first()
    chathistory = ChatHistory.query.filter_by(user_id = device_info.email, chat_id = chat_id).delete()
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500
