from dataclasses import dataclass, field
from typing import List


@dataclass
class MedicalEvent:
    date: str = ""
    provider: str = ""
    reason: str = ""
    page: int = 0


@dataclass
class Report:
    name: str = ""
    ssn: str = ""
    claim_title: str = ""
    dli: str = ""
    aod: str = ""
    date_of_birth: str = ""
    age_at_aod: int = 0
    current_age: int = 0
    last_grade_completed: int = 0
    attended_special_ed_classes: str = ""
    medical_events: List[MedicalEvent] = field(default_factory=list)
