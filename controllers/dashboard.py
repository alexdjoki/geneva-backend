from flask import Blueprint, request, jsonify
from models.auth import AccessKey
from models.chat_history import ChatHistory
from models.product_history import ProductHistory

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET', 'POST', 'OPTIONS'])

def index():
    token_count = AccessKey.query.count()
    chat_count = ChatHistory.query.count()
    product_count = ProductHistory.query.count()
    return jsonify({"token_count": token_count, "chat_count": chat_count, "product_count": product_count, "monthly_active": 2459, "total_input_tokens": 125223, "total_output_tokens": 4231232, "cost_to_date": 0})