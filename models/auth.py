from models import BaseModel
from app import db

class AccessKey(BaseModel):
    __tablename__ = 'access_key'

    device_id = db.Column(db.Text, nullable=False)
    access_key = db.Column(db.Text, nullable=False)
    status = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'device_id': self.device_id,
            'access_key': self.access_key,
            'status': self.status,
        }