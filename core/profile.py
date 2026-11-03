from dataclasses import dataclass, fields


@dataclass(frozen=True)
class StudentProfile:
    """Facts about a student that decide which documents and stations apply.

    Rule packs refer to these field names in their ``when`` criteria, so adding a
    field here makes it usable in rules without touching the engine.
    """

    category: str = "OPEN"          # OPEN, OBC-NCL, EWS, SC, ST
    home_state: str = ""            # state of domicile, e.g. "Madhya Pradesh"
    admission_route: str = "JoSAA"  # JoSAA, CSAB, DASA
    pwd: bool = False               # person with disability
    hostel_required: bool = True    # staying in the hostel

    def __post_init__(self):
        object.__setattr__(self, "category", _clean(self.category, "category"))
        object.__setattr__(self, "admission_route", _clean(self.admission_route, "admission_route"))
        object.__setattr__(self, "home_state", (self.home_state or "").strip())
        object.__setattr__(self, "pwd", bool(self.pwd))
        object.__setattr__(self, "hostel_required", bool(self.hostel_required))

    @classmethod
    def field_names(cls) -> set:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        unknown = set(data) - cls.field_names()
        if unknown:
            raise ValueError(f"Unknown profile fields: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _clean(value: str, name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"Profile field '{name}' must not be empty")
    return value
