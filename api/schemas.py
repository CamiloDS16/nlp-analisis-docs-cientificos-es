from pydantic import BaseModel


class Task1Prediction(BaseModel):
    label: str
    label_name: str
    confidence: float
    explanation: str


class Task2Prediction(BaseModel):
    is_contribution: bool
    label: str
    confidence: float
    evidence: str


class Segment(BaseModel):
    id: str
    position: int
    text: str
    task1: Task1Prediction
    task2: Task2Prediction


class ModelInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    latency_ms: int
    relative_cost: str


class Contribution(BaseModel):
    id: str
    label: str
    text: str


class Summary(BaseModel):
    dominant_label: str
    dominant_label_name: str
    contribution_count: int
    contributions: list[Contribution]
    label_distribution: dict[str, int]


class AnalyzeRequest(BaseModel):
    text: str
    task1_model_id: str = "commercial-api-gemini-task1"
    task2_model_id: str = "commercial-api-gemini-task2"


class AnalyzeResponse(BaseModel):
    input: dict
    models: dict[str, ModelInfo]
    segments: list[Segment]
    summary: Summary


class CompareRequest(BaseModel):
    text: str
    task1_model_id: str = "commercial-api-gemini-task1"
