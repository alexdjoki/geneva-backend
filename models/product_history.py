from models import BaseModel
from app import db

class ProductHistory(BaseModel):
    __tablename__ = 'product_history'

    user_id = db.Column(db.Text, nullable=False)
    search = db.Column(db.Text, nullable=False)
    products = db.Column(db.Text, nullable=False)
    product_type = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'search': self.search,
            'products': self.products,
            'product_type': self.product_type,
            'created_at': self.created_at.isoformat()
        }