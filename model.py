from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import BYTEA

db = SQLAlchemy()

class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    person = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)  # store NumPy as binary
