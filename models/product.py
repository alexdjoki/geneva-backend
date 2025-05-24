from models import BaseModel
from app import db

class Product(BaseModel):
    __tablename__ = 'products'

    image = db.Column(db.Text, nullable=False)
    price = db.Column(db.Text, nullable=False)
    url = db.Column(db.Text, nullable=False)
    category = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'image': self.image,
            'price': self.price,
            'url': self.url,
            'category': self.category
        }