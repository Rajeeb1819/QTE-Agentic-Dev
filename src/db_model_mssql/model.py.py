
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, DateTime, JSON, CheckConstraint
from datetime import datetime, timezone

class Base(DeclarativeBase):
    pass


class ApplicationChatHistory(Base):
    __tablename__ = "app_chat_history"
    __table_args__ = {"schema": "dbo"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))       # user, agent, critic, etc.
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ApplicationShortMemory(Base):
    __tablename__ = "app_short_memory"
    __table_args__ = {"schema": "dbo"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )



# db/models.py (append)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone

class ApplicationTeamState(Base):
    # __tablename__ = "app_team_state"
    # __table_args__ = {"schema": "dbo"}

    # id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # application_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # updated_at: Mapped[datetime] = mapped_column(
    #     DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    # )
    
    __tablename__ = "app_team_state"
    __table_args__ = (
        CheckConstraint("ISJSON([state]) = 1", name="ck_app_team_state_isjson"),
        {"schema": "dbo"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    state: Mapped[dict] = mapped_column(JSON, nullable=False)  # stored as NVARCHAR in MSSQL
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

