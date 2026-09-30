from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOLO_TRAINER_", arbitrary_types_allowed=True)

    workspace_root: Path = Field(default=REPO_ROOT / "workspace")
    jwt_secret: str = Field(default="yolo-trainer-dev-secret-change-me")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60 * 24 * 7)
    default_admin_username: str = Field(default="admin")
    default_admin_password: str = Field(default="admin123")

    @property
    def database_path(self) -> Path:
        return self.workspace_root / "app.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    def ensure_workspace(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_workspace()
    return settings
