from models import BaseModel
from app import db

class AccessKey(BaseModel):
    __tablename__ = 'access_key'

    device_id = db.Column(db.Text, nullable=False)
    access_key = db.Column(db.Text, nullable=False)
    status = db.Column(db.Integer, nullable=False)
    email = db.Column(db.Text, nullable=False)
    valid_date = db.Column(db.DateTime, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'device_id': self.device_id,
            'access_key': self.access_key,
            'status': self.status,
            'email': self.email,
            'valid_date': self.valid_date,
            'count': self.count
        }