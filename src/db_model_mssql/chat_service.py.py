# src/db_model_mssql/chat_service.py
from src.db_model_mssql.chat_repository import ChatRepository, TeamStateRepository
from src.db_model_mssql.session import get_async_session  # your AsyncSession factory

repo = ChatRepository()
_team_repo = TeamStateRepository()


class ChatService:
    async def store_user_message(self, application_id: str, content: str):
        """
        Store the user's message in both history and short-term memory.
        """
        async with get_async_session() as session:
            await repo.add_history(session, application_id, "user", content)
            await repo.add_short_memory(session, application_id, "user", content)

    async def store_agent_message(self, application_id: str, role: str, content: str):
        """
        Store any non-user agent message to FULL history ONLY.
        Do NOT write to STM here (we only put final assistant into STM).
        """
        async with get_async_session() as session:
            await repo.add_history(session, application_id, role, content)

    async def store_final_message(self, application_id: str, role: str, content: str):
        """
        Store the final assistant message to short-term memory ONLY.
        History for this message is already handled via persist_run_messages().
        """
        async with get_async_session() as session:
            await repo.add_short_memory(session, application_id, role, content)

    async def load_short_memory(self, application_id: str):
        async with get_async_session() as session:
            return await repo.get_short_memory(session, application_id)

    async def delete_short_memory(self, application_id: str):
        async with get_async_session() as session:
            return await repo.delete_short_memory(session, application_id)

    async def load_full_history(self, application_id: str):
        async with get_async_session() as session:
            return await repo.get_full_history(session, application_id)


class TeamStateService:
    async def save(self, application_id: str, state: dict):
        async with get_async_session() as session:
            await _team_repo.upsert_state(session, application_id, state)

    async def load(self, application_id: str):
        async with get_async_session() as session:
            return await _team_repo.get_state(session, application_id)


team_state_service = TeamStateService()