# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, text
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))

    # Quan hệ để sau này query user.chats hoặc user.interactions dễ dàng
    chats = relationship("ChatHistory", back_populates="owner")
    interactions = relationship("UserInteraction", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("public.users.id"))
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))

    owner = relationship("User", back_populates="chats")

# Bảng này cực kỳ quan trọng để thu thập hành vi ngầm cho khóa luận
class UserInteraction(Base):
    __tablename__ = "user_interactions"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("public.users.id"))
    article_id = Column(Integer) # ID bài báo từ bảng raw_data
    action = Column(String(50))    # VD: 'click', 'read_more', 'summary'
    created_at = Column(DateTime, server_default=text("now()"))

    user = relationship("User", back_populates="interactions")