import json
from sqlalchemy import create_engine, Column, String, Integer, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class TraceRecord(Base):
    __tablename__ = 'execution_traces'
    id = Column(String, primary_key=True)
    task_type = Column(String)
    query_text = Column(Text)
    final_result = Column(Text)
    confidence = Column(Float)
    full_trace_json = Column(Text)

class StorageService:
    def __init__(self, db_url="sqlite:///./satquery.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_trace(self, trace_id: str, task_type: str, query: str, result: str, confidence: float, trace_data: dict):
        with self.Session() as session:
            record = TraceRecord(
                id=trace_id,
                task_type=task_type,
                query_text=query,
                final_result=result,
                confidence=confidence,
                full_trace_json=json.dumps(trace_data)
            )
            session.add(record)
            session.commit()
