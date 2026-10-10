"""Core package for Autonomous Continual Learning LLM System."""

from core.config import settings
from core.hardware import get_device_strategy, DeviceStrategy
from core.schemas import (
    UserQuerySchema,
    ScrapedDocumentSchema,
    CleanedDocumentSchema,
    ChatMessageSchema,
    TrainingSampleSchema,
    PreferencePairSchema,
    BenchmarkCaseSchema,
    EvaluationReportSchema,
)

__all__ = [
    "settings",
    "get_device_strategy",
    "DeviceStrategy",
    "UserQuerySchema",
    "ScrapedDocumentSchema",
    "CleanedDocumentSchema",
    "ChatMessageSchema",
    "TrainingSampleSchema",
    "PreferencePairSchema",
    "BenchmarkCaseSchema",
    "EvaluationReportSchema",
]
