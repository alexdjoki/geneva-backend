from models import BaseModel
from app import db

class Devices(BaseModel):
    __tablename__ = 'devices'

    device_id = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'device_id': self.device_id,
            'email': self.email,
        }