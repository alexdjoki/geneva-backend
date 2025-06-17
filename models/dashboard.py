from models import BaseModel
from app import db

class Dashboard(BaseModel):
    __tablename__ = 'dashboard'

    monthly_active = db.Column(db.Integer, nullable=False)
    token_count = db.Column(db.Integer, nullable=False)
    total_input_tokens = db.Column(db.Integer, nullable=False)
    total_output_tokens = db.Column(db.Integer, nullable=False)
    cost_to_date = db.Column(db.Integer, nullable=False)
    chat_count = db.Column(db.Integer, nullable=False)
    product_count = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'monthly_active': self.monthly_active,
            'token_count': self.token_count,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'cost_to_date': self.cost_to_date,
            'chat_count': self.chat_count,
            'product_count': self.product_count,
        }