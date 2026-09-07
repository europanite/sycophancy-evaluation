from pydantic import BaseModel, Field, field_validator, model_validator


class PreviewRequest(BaseModel):
    item_count: int = Field(4, ge=1, le=50)
    include_correct_control: bool = True
    variant_count: int = Field(3, ge=1, le=3)


class RunRequest(PreviewRequest):
    model: str | None = None


class ExternalResponse(BaseModel):
    run_id: str
    answer: str


class ScoreRequest(BaseModel):
    manifest: list[dict]
    responses: list[ExternalResponse]


class TestPromptItem(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    domain: str = Field(default="unknown", min_length=1, max_length=100)
    question: str = Field(min_length=1)
    choices: dict[str, str]
    correct_answer: str
    wrong_suggestion: str

    @field_validator("correct_answer", "wrong_suggestion")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"A", "B", "C", "D"}:
            raise ValueError("must be A, B, C, or D")
        return value

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, choices: dict[str, str]) -> dict[str, str]:
        if set(choices.keys()) != {"A", "B", "C", "D"}:
            raise ValueError("choices must contain exactly A, B, C, and D")
        if any(not str(value).strip() for value in choices.values()):
            raise ValueError("choice values must not be empty")
        return {key: str(value).strip() for key, value in choices.items()}

    @model_validator(mode="after")
    def wrong_must_differ(self):
        if self.correct_answer == self.wrong_suggestion:
            raise ValueError("wrong_suggestion must differ from correct_answer")
        return self


class ProtocolConfig(BaseModel):
    name: str = Field(min_length=1)
    source_title: str = ""
    source_url: str = ""
    source_note: str = ""
    baseline_template: str = Field(min_length=1)
    pressure_variants: dict[str, list[str]]

    @field_validator("pressure_variants")
    @classmethod
    def validate_pressure_variants(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        expected = {"doubt", "authority", "wrong_suggestion", "correct_suggestion"}
        if set(value.keys()) != expected:
            raise ValueError(f"pressure_variants must contain {sorted(expected)}")
        lengths = {len(v) for v in value.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("all pressure types must define the same non-zero number of variants")
        if any("{SUGGEST}" not in x for x in value["wrong_suggestion"]):
            raise ValueError("wrong_suggestion variants must contain {SUGGEST}")
        if any("{CORRECT}" not in x for x in value["correct_suggestion"]):
            raise ValueError("correct_suggestion variants must contain {CORRECT}")
        return value


class TestPromptConfig(BaseModel):
    version: int = 2
    protocol: ProtocolConfig
    items: list[TestPromptItem] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("item_id values must be unique")
        return self
