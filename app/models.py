from datetime import datetime

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    email = Column(String)
    access_token = Column(String)
    refresh_token = Column(String)
    token_expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    tracks = relationship("Track", back_populates="user", cascade="all, delete-orphan")
    listening_history = relationship("ListeningHistory", back_populates="user", cascade="all, delete-orphan")
    clusters = relationship("Cluster", back_populates="user", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    spotify_track_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    artist = Column(String)
    album = Column(String)
    duration_ms = Column(Integer)
    popularity = Column(Integer)
    genres = Column(String)
    lastfm_tags = Column(String)
    added_at = Column(DateTime)

    __table_args__ = (UniqueConstraint("user_id", "spotify_track_id"),)

    user = relationship("User", back_populates="tracks")
    listening_history = relationship("ListeningHistory", back_populates="track", cascade="all, delete-orphan")


class ListeningHistory(Base):
    __tablename__ = "listening_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    played_at = Column(DateTime, nullable=False, index=True)

    __table_args__ = (UniqueConstraint("user_id", "track_id", "played_at"),)

    user = relationship("User", back_populates="listening_history")
    track = relationship("Track", back_populates="listening_history")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    cluster_index = Column(Integer, nullable=False)
    label = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user_id", "cluster_index"),)

    user = relationship("User", back_populates="clusters")
