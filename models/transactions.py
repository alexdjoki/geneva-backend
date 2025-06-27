from models import BaseModel
from app import db

class Transactions(BaseModel):
    __tablename__ = 'transactions'

    userid = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, nullable=False)
    wallet_address = db.Column(db.Text, nullable=False)
    valid_date = db.Column(db.DateTime, nullable=False)
    access_key = db.Column(db.Text, nullable=False)
    amount = db.Column(db.Text, nullable=False)
    order_id = db.Column(db.Text, nullable=False)
    status = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'userid': self.userid,
            'wallet_address': self.wallet_address,
            'email': self.email,
            'valid_date': self.valid_date,
            'access_key': self.access_key,
            'order_id': self.order_id,
            'status': self.status,
        }