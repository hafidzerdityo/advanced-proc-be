from repositories.postgres.config.db_config import AsyncSessionLocal
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import AsyncGenerator

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
