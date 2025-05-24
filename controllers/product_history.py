from flask import Blueprint, request, jsonify
from models.product_history import ProductHistory
from app import db
from datetime import datetime

product_history_bp = Blueprint('product-history', __name__)

@product_history_bp.route('/list', methods=['POST'])

def list():
  data = request.get_json()
  user_id = data.get('user_id')
  try:
    producthistory = ProductHistory.query.all()
    list = [item.to_dict() for item in producthistory]
    return jsonify(list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@product_history_bp.route('/delete', methods=['POST'])

def delete():
  data = request.get_json()
  user_id = data.get('user_id')
  product_id = data.get('id')
  try:
    producthistory = ProductHistory.query.filter_by(user_id = user_id, id = product_id).delete()
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500