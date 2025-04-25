from models import BaseModel
from app import db

class ChatHistory(BaseModel):
    __tablename__ = 'chat_history'

    user_id = db.Column(db.Integer, nullable=False)
    chat_id = db.Column(db.Integer, nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    status_report = db.Column(db.Text, nullable=False)
    opinion = db.Column(db.Text, nullable=False)
    level = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'chat_id': self.chat_id,
            'question': self.question,
            'answer': self.answer,
            'status_report': self.status_report,
            'opinion': self.opinion,
            'level': self.level,
            'created_at': self.created_at.isoformat()
        }