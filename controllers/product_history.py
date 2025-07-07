from flask import Blueprint, request, jsonify
from models.product_history import ProductHistory
from models.devices import Devices
from app import db
from datetime import datetime

product_history_bp = Blueprint('product-history', __name__)

@product_history_bp.route('/product-content', methods=['POST'])

def product_content():
  data = request.get_json()
  id = data.get('id')
  try:  
    product = ProductHistory.query.filter_by(id = id).first()
    return jsonify(product.products), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@product_history_bp.route('/list', methods=['POST'])

def list():
  data = request.get_json()
  device_id = data.get('user_id')
  try:
    device_info = Devices.query.filter_by(device_id = device_id).first()
    producthistory = ProductHistory.query.filter_by(user_id = device_info.email).all()
    result_list = [{'id': item.id, 'query': item.search, 'product_type': item.product_type} for item in producthistory]
    return jsonify(result_list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500

@product_history_bp.route('/delete', methods=['POST'])

def delete():
  data = request.get_json()
  device_id = data.get('user_id')
  product_id = data.get('id')
  try:
    device_info = Devices.query.filter_by(device_id = device_id).first()
    producthistory = ProductHistory.query.filter_by(user_id = device_info.email, id = product_id).delete()
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500