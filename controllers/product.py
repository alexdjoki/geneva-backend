from flask import Blueprint, request, jsonify
from models.product import Product
from app import db
from datetime import datetime
import random

product_bp = Blueprint('product', __name__)

@product_bp.route('/list', methods=['POST'])

def list():
  data = request.get_json()
  category = data.get('category')
  try:
    if category == 'random':
      products = Product.query.all()
      products = random.sample(products, min(40, len(products)))
    else:
      products = Product.query.filter_by(category = category).all()
    list = [item.to_dict() for item in products]
    return jsonify(list), 200
  except Exception as e:
    print(e)
    return jsonify({'message': str(e)}), 500