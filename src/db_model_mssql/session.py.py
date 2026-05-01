# db/config.py
import os
from dotenv import load_dotenv
from sqlalchemy import URL
# from .engine import engine
from src.db_model_mssql.model import Base
load_dotenv()

# Prefer env vars; fall back to defaults in dev
MSSQL_USER = os.getenv("MSSQL_USER", "user")
MSSQL_PASS = os.getenv("MSSQL_PASS", "pass")
MSSQL_HOST = os.getenv("MSSQL_HOST", "localhost")  # e.g. yourserver.database.windows.net
MSSQL_PORT = int(os.getenv("MSSQL_PORT", "1433"))
MSSQL_DB   = os.getenv("MSSQL_DB", "yourdb")
MSSQL_DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")

# Structured URL (drivername 'mssql+aioodbc' for async ORM)
# SQLAlchemy supports aioodbc for SQL Server dialect. [1](https://www.sqlalchemy.org/docs/21/dialects/mssql.html)
# Azure SQL commonly requires Encrypt=yes and TrustServerCertificate=no with ODBC Driver 18. [2](https://learn.microsoft.com/en-us/azure/azure-sql/database/azure-sql-python-quickstart?view=azuresql)
ASYNC_DB_URL = URL.create(
    drivername="mssql+aioodbc",
    username=MSSQL_USER,
    password=MSSQL_PASS,
    host=MSSQL_HOST,
    port=MSSQL_PORT,
    database=MSSQL_DB,
    query={
        "driver": MSSQL_DRIVER,
        "Encrypt": "yes",
        "TrustServerCertificate": "no",
        "Connection Timeout": "30",
    },
)


# db/engine.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# from .config import ASYNC_DB_URL

# Create the async engine; tune pool sizes to your needs
engine = create_async_engine(
    ASYNC_DB_URL,
    echo=False,          # set True to see SQL in logs
    future=True,         # SQLAlchemy 2.0 style
    pool_size=5,         # similar to max_size
    max_overflow=10,     # burst capacity
    pool_pre_ping=True,  # checks connections before using
)

# Session factory (scoped per request/task)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

from contextlib import asynccontextmanager
# from .engine import AsyncSessionLocal

@asynccontextmanager
async def get_async_session():
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()




async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)



from sqlalchemy import text
# from src.db.engine import engine

async def ping():
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        print(result.scalar_one())

