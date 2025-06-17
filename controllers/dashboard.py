from flask import Blueprint, request, jsonify
# from models.auth import AccessKey
# from models.chat_history import ChatHistory
# from models.product_history import ProductHistory
from models.dashboard import Dashboard

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET', 'POST', 'OPTIONS'])

def index():
    data = Dashboard.query.all()
    data_list = [item.to_dict() for item in data]
    return jsonify(data_list[0])
