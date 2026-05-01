
# src/db_models/scope_service.py
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.db_model_mssql.session import get_async_session

@dataclass
class ScopeContext:
    application_id: str
    jira_projectkey: Optional[str] = None
    aha_id: Optional[str] = None
    confluence_spacekey: Optional[str] = None

    def all_ids_lower(self) -> "ScopeContext":
        return ScopeContext(
            application_id=self.application_id,
            jira_projectkey=(self.jira_projectkey.lower() if self.jira_projectkey else None),
            aha_id=(self.aha_id.lower() if self.aha_id else None),
            confluence_spacekey=(self.confluence_spacekey.lower() if self.confluence_spacekey else None),
        )

class ScopeService:
    async def get_scope(self, application_id: str) -> ScopeContext:
        async with get_async_session() as session:
            return await self._load_scope(session, application_id)

    async def _load_scope(self, session: AsyncSession, application_id: str) -> ScopeContext:
        # One-pass pivot to fetch single values for JIRA, AHA, CONFLUENCE
        query = text("""
        SELECT TOP (1)
            AO.appid AS application_id,
            AO.confluence AS confluence_spacekey,
            MAX(CASE WHEN UPPER(T.name) = 'JIRA' THEN JK.tool_id END) AS jira_projectkey,
            MAX(CASE WHEN UPPER(T.name) = 'AHA'  THEN JK.tool_id END) AS aha_id
        FROM [applicationonboarding$applicationonboarding] AO
        JOIN [applicationonboarding$applicationonboardingtool] BC
          ON BC.applicationonboarding$applicationonboardingtool_applicationonboarding = AO.id
        JOIN [applicationonboarding$applicationonboardingtool_keyid] CK
          ON CK.applicationonboarding$applicationonboardingtoolid = BC.id
        JOIN [applicationonboarding$keyid] JK
          ON JK.id = CK.applicationonboarding$keyidid
        JOIN [applicationonboarding$tool] T
          ON T.id = JK.applicationonboarding$keyid_tool
        WHERE AO.appid = :appid
        GROUP BY AO.appid, AO.confluence
        ORDER BY AO.appid
    """)
        appid_param = int(application_id)
        row = (await session.execute(query, {"appid": appid_param})).mappings().fetchone()

        if not row:
            raise ValueError(f"Scope not found for {application_id}")

        return ScopeContext(
            application_id=str(row["application_id"]),
            jira_projectkey=row.get("jira_projectkey"),
            aha_id=row.get("aha_id"),
            confluence_spacekey=row.get("confluence_spacekey"),
        )
