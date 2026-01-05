from __future__ import annotations
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Load .env file from backend directory (parent of app/core)
    _env_file_path = Path(__file__).parent.parent.parent / ".env"
    
    model_config = SettingsConfigDict(
        env_file=str(_env_file_path) if _env_file_path.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    app_name: str = Field(default="Interior Visual Search", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="interior-products", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_dim: int = Field(default=1024, alias="PINECONE_DIM")

    instance_model_name: str = Field(default="google/vit-base-patch16-224-in21k", alias="INSTANCE_MODEL_NAME")
    semantic_model_name: str = Field(default="openai/clip-vit-base-patch32", alias="SEMANTIC_MODEL_NAME")

    enable_pinecone_rerank: bool = Field(default=False, alias="ENABLE_PINECONE_RERANK")
    pinecone_rerank_model: str = Field(default="bge-reranker-v2-m3", alias="PINECONE_RERANK_MODEL")
    
    # RT-DETR Detection Model
    rtdetr_model_path: str = Field(
        default=r"D:\image_image_search\backend\app\models\rtdetr-x.pt",
        alias="RTDETR_MODEL_PATH"
    )
    
    # SAM2.1 Segmentation Model (Ultralytics)
    sam2_model_path: str = Field(
        default=r"D:\image_image_search\backend\app\models\sam2.1_l.pt",
        alias="SAM2_MODEL_PATH"
    )
    
    # Image validation settings
    image_min_dimension: int = Field(default=100, alias="IMAGE_MIN_DIMENSION")
    image_max_dimension: int = Field(default=4096, alias="IMAGE_MAX_DIMENSION")
    image_max_size_mb: int = Field(default=10, alias="IMAGE_MAX_SIZE_MB")
    
    # Image storage directories
    catalog_images_dir: str = Field(default="images/catalog", alias="CATALOG_IMAGES_DIR")
    search_images_dir: str = Field(default="images/search", alias="SEARCH_IMAGES_DIR")
    
    # Search settings
    search_candidate_multiplier: int = Field(default=15, alias="SEARCH_CANDIDATE_MULTIPLIER")
    search_deduplicate_skus: bool = Field(default=True, alias="SEARCH_DEDUPLICATE_SKUS")
    max_candidate_k: int = Field(default=5000, alias="MAX_CANDIDATE_K")
    
    # Advanced embedding settings
    enable_multiscale_embedding: bool = Field(default=True, alias="ENABLE_MULTISCALE_EMBEDDING")
    embedding_scales: str = Field(default="224,384,512", alias="EMBEDDING_SCALES")
    enable_rotation_augmentation: bool = Field(default=True, alias="ENABLE_ROTATION_AUGMENTATION")
    
    # Advanced reranking settings
    enable_multistage_rerank: bool = Field(default=True, alias="ENABLE_MULTISTAGE_RERANK")
    rerank_stage1_ratio: float = Field(default=0.3, alias="RERANK_STAGE1_RATIO")