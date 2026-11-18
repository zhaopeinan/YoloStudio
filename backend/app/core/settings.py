from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOLOSTUDIO_", arbitrary_types_allowed=True)

    workspace_root: Path = Field(default=REPO_ROOT / "workspace")
    jwt_secret: str = Field(default="yolostudio-dev-secret-change-me")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60 * 24 * 7)
    default_admin_username: str = Field(default="admin")
    default_admin_password: str = Field(default="admin123")
    rknn_python: Path | None = Field(
        default=None,
        description="Python executable that has rknn-toolkit2 installed",
    )
    rknn_target_platform: str = Field(default="rk3588")
    rknn_quant_mode: str = Field(
        default="fp16",
        description=(
            "RKNN quant mode: fp16 (default, avoids YOLOv8 score collapse), "
            "hybrid (INT8 body + FP16 outputs), or int8 (often breaks class scores)"
        ),
    )
    remote_enabled: bool = Field(default=True, description="Enable GPU05 remote training/sync")
    remote_ssh_host: str = Field(
        default="gpu05",
        description="SSH host alias or address for GPU05 (uses ~/.ssh/config)",
    )
    remote_root: Path = Field(
        default=Path("/data/home/pnzhao/yolostudio-remote"),
        description="Fixed remote workspace root on GPU05 (project data only)",
    )
    remote_python: str = Field(
        default="python3",
        description="Python executable on GPU05 with ultralytics/torch available",
    )
    remote_connect_timeout: int = Field(default=15, ge=1, le=120)
    remote_host_id: str = Field(default="gpu05", description="Logical host id used by the UI/API")
    remote_host_label: str = Field(default="GPU05 (V100)")

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
