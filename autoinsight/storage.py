from datetime import datetime, timezone
from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, Table, create_engine, select

metadata = MetaData()
events = Table("predictions", metadata, Column("id", Integer, primary_key=True),
               Column("created", DateTime(timezone=True), nullable=False),
               Column("car", JSON, nullable=False), Column("result", JSON, nullable=False))


class History:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        metadata.create_all(self.engine)

    def append(self, car, result):
        with self.engine.begin() as conn:
            conn.execute(events.insert().values(created=datetime.now(timezone.utc), car=car, result=result))

    def recent(self, limit):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(
                select(events).order_by(events.c.id.desc()).limit(limit)).mappings()]
