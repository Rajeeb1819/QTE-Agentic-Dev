# src/db_model_mssql/chat_repository.py
import asyncio
import logging
from typing import List, Dict, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from src.db_model_mssql.model import (
    ApplicationChatHistory,
    ApplicationShortMemory,
    ApplicationTeamState,
)

SHORT_WINDOW = 20
logger = logging.getLogger(__name__)


async def retry_on_deadlock(func, *args, max_retries=3, **kwargs):
    """Retry database operations on deadlock (SQL error 40001)"""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except DBAPIError as e:
            # Check if it's a deadlock error (SQL error code 40001)
            error_code = getattr(e.orig, 'args', [None])[0]
            if error_code == '40001' and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.1  # 0.1s, 0.2s, 0.4s
                logger.warning(f"Deadlock detected, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            raise


class ChatRepository:
    async def add_history(self, session: AsyncSession, application_id: str, role: str, content: str):
        """
        Append a row into the full conversation history.
        """
        async with session.begin():
            session.add(
                ApplicationChatHistory(
                    application_id=application_id,
                    role=role,
                    content=content,
                )
            )

    async def add_short_memory(self, session: AsyncSession, application_id: str, role: str, content: str):
        """
        Append a row into the short-term memory table and prune to SHORT_WINDOW
        most recent rows for this application_id.
        Includes retry logic for deadlock handling during concurrent access.
        """
        async def _execute_with_cleanup():
            async with session.begin():
                entry = ApplicationShortMemory(
                    application_id=application_id,
                    role=role,
                    content=content
                )
                session.add(entry)
                await session.flush()

                # Keep only last SHORT_WINDOW by created_at desc
                keep_ids_subq = (
                    select(ApplicationShortMemory.id)
                    .where(ApplicationShortMemory.application_id == application_id)
                    .order_by(ApplicationShortMemory.created_at.desc())
                    .limit(SHORT_WINDOW)
                )

                del_stmt = (
                    delete(ApplicationShortMemory)
                    .where(ApplicationShortMemory.application_id == application_id)
                    .where(ApplicationShortMemory.id.notin_(keep_ids_subq))
                )
                await session.execute(del_stmt)
        
        # Retry up to 3 times on deadlock
        await retry_on_deadlock(_execute_with_cleanup, max_retries=3)

    async def get_short_memory(self, session: AsyncSession, application_id: str) -> List[Dict]:
        """
        Return STM rows ascending by created_at, limited to SHORT_WINDOW. Includes id and created_at for logging.
        """
        q = (
            select(ApplicationShortMemory)
            .where(ApplicationShortMemory.application_id == application_id)
            .order_by(ApplicationShortMemory.created_at.desc())
            .limit(SHORT_WINDOW)
        )
        rows_desc = (await session.execute(q)).scalars().all()
        rows = list(reversed(rows_desc))  # oldest -> newest
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    async def delete_short_memory(self,session: AsyncSession, application_id: str):
        try:
            q = (
                delete(ApplicationShortMemory)
                .where(ApplicationShortMemory.application_id == application_id))
            result = await session.execute(q)
            await session.commit()
            return result.rowcount   #number of deleted rows
        except SQLAlchemyError as e:
            await session.rollback()
            raise  # re-raise the exception after rollback

    async def get_full_history(self, session: AsyncSession, application_id: str) -> List[Dict]:
        q = (
            select(ApplicationChatHistory)
            .where(ApplicationChatHistory.application_id == application_id)
            .order_by(ApplicationChatHistory.created_at.asc())
        )
        rows = (await session.execute(q)).scalars().all()
        return [
            {"role": r.role, "content": r.content, "created_at": r.created_at}
            for r in rows
        ]


class TeamStateRepository:
    async def upsert_state(self, session: AsyncSession, application_id: str, state: dict):
        async with session.begin():
            q = select(ApplicationTeamState).where(
                ApplicationTeamState.application_id == application_id
            )
            obj = (await session.execute(q)).scalars().first()
            if obj:
                obj.state = state
            else:
                session.add(
                    ApplicationTeamState(application_id=application_id, state=state)
                )

    async def get_state(self, session: AsyncSession, application_id: str) -> Optional[dict]:
        q = select(ApplicationTeamState).where(
            ApplicationTeamState.application_id == application_id
        )
        obj = (await session.execute(q)).scalars().first()
        return obj.state if obj else None