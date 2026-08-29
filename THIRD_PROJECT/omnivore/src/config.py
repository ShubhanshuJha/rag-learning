"""
Typed configuration loader for Omnivore.

Reads omnivore/config.yaml into frozen dataclasses, so a typo in the YAML
fails at startup with a precise message instead of surfacing as an
AttributeError somewhere deep in a request.

Usage:
    from config import CONFIG
    CONFIG.embedding.model        # "all-minilm"
    CONFIG.data.dir               # absolute Path, resolved

Path handling:
    Relative paths in the YAML resolve against the config file's own directory,
    not the working directory. That is what lets main.py run from anywhere,
    unlike app.py, whose hardcoded '../data' still depends on the CWD.

Overrides:
    OMNIVORE_CONFIG   absolute path to an alternative YAML file
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# config.py lives in omnivore/src/, so the default config is one level up.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

VALID_BACKENDS = {"ollama", "groq"}
VALID_MODES = {"skip", "replace", "append"}
VALID_METRICS = {"cosine", "l2", "ip"}


class ConfigError(ValueError):
    """Raised when the YAML is missing, malformed, or semantically invalid."""


# =========================================================================
# Section types
# =========================================================================


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True


@dataclass(frozen=True)
class DataConfig:
    dir: Path = Path("data")


@dataclass(frozen=True)
class VectorStoreConfig:
    collection: str = "omnivore_docs"
    persist_directory: Path = Path("data/vector_store")
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "all-minilm"
    use_ollama: bool = True
    batch_size: int = 256
    max_retries: int = 3
    retry_delay: float = 2.0
    batch_delay: float = 0.3

    @property
    def call_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for EmbeddingManager.generate_embeddings."""
        return {
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "batch_delay": self.batch_delay,
        }


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 200


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "ollama"
    temperature: float = 0.1
    max_tokens: int = 1024
    ollama_model: str = "gemma2:9b"
    groq_model: str = "gemma2-9b-it"
    api_key_env: str = "GROQ_API_KEY"

    @property
    def model(self) -> str:
        """The model name for whichever backend is selected."""
        return self.ollama_model if self.backend == "ollama" else self.groq_model

    @property
    def api_key(self) -> Optional[str]:
        """Resolve the Groq key from the environment; None for Ollama."""
        return None if self.backend == "ollama" else os.getenv(self.api_key_env)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.1
    summarize: bool = False


@dataclass(frozen=True)
class IngestionConfig:
    ingest_on_startup: bool = True
    default_mode: str = "skip"
    max_upload_mb: int = 100
    upload_subdir: str = "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@dataclass(frozen=True)
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    source_path: Optional[Path] = None

    @property
    def upload_dir(self) -> Path:
        return self.data.dir / self.ingestion.upload_subdir


# =========================================================================
# Parsing
# =========================================================================


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Fetch a top-level section, tolerating absence and explicit nulls."""
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(section: Dict[str, Any], known: set, name: str) -> None:
    """Fail loudly on unrecognized keys — almost always a typo."""
    unknown = set(section) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{name}': {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )


def _resolve(base: Path, value: Any) -> Path:
    """Resolve a possibly-relative path against the config file's directory."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate(cfg: Config) -> None:
    """Semantic checks that a type annotation cannot express."""
    problems = []

    if cfg.chunking.chunk_overlap >= cfg.chunking.chunk_size:
        problems.append(
            f"chunking.chunk_overlap ({cfg.chunking.chunk_overlap}) must be less than "
            f"chunk_size ({cfg.chunking.chunk_size})."
        )
    if cfg.chunking.chunk_size < 1:
        problems.append("chunking.chunk_size must be at least 1.")
    if cfg.chunking.chunk_overlap < 0:
        problems.append("chunking.chunk_overlap cannot be negative.")
    if cfg.embedding.batch_size < 1:
        problems.append("embedding.batch_size must be at least 1.")
    if cfg.embedding.max_retries < 1:
        problems.append("embedding.max_retries must be at least 1.")
    if cfg.embedding.retry_delay < 0:
        problems.append("embedding.retry_delay cannot be negative.")
    if cfg.embedding.batch_delay < 0:
        problems.append("embedding.batch_delay cannot be negative.")
    if cfg.vector_store.distance_metric not in VALID_METRICS:
        problems.append(
            f"vector_store.distance_metric must be one of {sorted(VALID_METRICS)}, "
            f"got '{cfg.vector_store.distance_metric}'."
        )
    if cfg.llm.backend not in VALID_BACKENDS:
        problems.append(
            f"llm.backend must be one of {sorted(VALID_BACKENDS)}, got '{cfg.llm.backend}'."
        )
    if cfg.llm.temperature < 0:
        problems.append("llm.temperature cannot be negative.")
    if cfg.llm.max_tokens < 1:
        problems.append("llm.max_tokens must be at least 1.")
    if cfg.retrieval.top_k < 1:
        problems.append("retrieval.top_k must be at least 1.")
    if not 0.0 <= cfg.retrieval.min_score <= 1.0:
        problems.append(
            f"retrieval.min_score must be between 0 and 1, got {cfg.retrieval.min_score}."
        )
    if cfg.ingestion.default_mode not in VALID_MODES:
        problems.append(
            f"ingestion.default_mode must be one of {sorted(VALID_MODES)}, "
            f"got '{cfg.ingestion.default_mode}'."
        )
    if cfg.ingestion.max_upload_mb < 1:
        problems.append("ingestion.max_upload_mb must be at least 1.")
    if not 1 <= cfg.app.port <= 65535:
        problems.append(f"app.port must be between 1 and 65535, got {cfg.app.port}.")

    # A misconfigured Groq backend fails at the first query rather than at
    # startup, which is a much worse place to discover it.
    if cfg.llm.backend == "groq" and not cfg.llm.api_key:
        problems.append(
            f"llm.backend is 'groq' but ${cfg.llm.api_key_env} is not set. "
            "Add it to .env or switch llm.backend to 'ollama'."
        )

    if problems:
        raise ConfigError(
            "Invalid configuration in {}:\n  - {}".format(
                cfg.source_path or "<defaults>", "\n  - ".join(problems)
            )
        )


def load_config(path: Optional[Path] = None) -> Config:
    """
    Read, parse, and validate the YAML. Raises ConfigError on any problem.

    Resolution order: explicit `path` argument, then $OMNIVORE_CONFIG, then
    omnivore/config.yaml.
    """
    if path is None:
        env_path = os.getenv("OMNIVORE_CONFIG")
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}\n"
            "Create it, or point OMNIVORE_CONFIG at an existing file."
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as ex:
        raise ConfigError(f"Could not parse {path}: {ex}") from ex

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a top-level mapping.")

    _reject_unknown(
        raw,
        {"app", "data", "vector_store", "embedding", "chunking", "llm",
         "retrieval", "ingestion"},
        "<top level>",
    )

    base = path.parent

    app_raw = _section(raw, "app")
    _reject_unknown(app_raw, {"host", "port", "debug"}, "app")

    data_raw = _section(raw, "data")
    _reject_unknown(data_raw, {"dir"}, "data")

    store_raw = _section(raw, "vector_store")
    _reject_unknown(
        store_raw, {"collection", "persist_directory", "distance_metric"}, "vector_store"
    )

    embed_raw = _section(raw, "embedding")
    _reject_unknown(
        embed_raw,
        {"model", "use_ollama", "batch_size", "max_retries", "retry_delay", "batch_delay"},
        "embedding",
    )

    chunk_raw = _section(raw, "chunking")
    _reject_unknown(chunk_raw, {"chunk_size", "chunk_overlap"}, "chunking")

    llm_raw = _section(raw, "llm")
    _reject_unknown(
        llm_raw, {"backend", "temperature", "max_tokens", "ollama", "groq"}, "llm"
    )
    ollama_raw = llm_raw.get("ollama") or {}
    groq_raw = llm_raw.get("groq") or {}
    _reject_unknown(ollama_raw, {"model"}, "llm.ollama")
    _reject_unknown(groq_raw, {"model", "api_key_env"}, "llm.groq")

    retrieval_raw = _section(raw, "retrieval")
    _reject_unknown(retrieval_raw, {"top_k", "min_score", "summarize"}, "retrieval")

    ingest_raw = _section(raw, "ingestion")
    _reject_unknown(
        ingest_raw,
        {"ingest_on_startup", "default_mode", "max_upload_mb", "upload_subdir"},
        "ingestion",
    )

    defaults = Config()
    try:
        cfg = Config(
            app=AppConfig(
                host=str(app_raw.get("host", defaults.app.host)),
                port=int(app_raw.get("port", defaults.app.port)),
                debug=bool(app_raw.get("debug", defaults.app.debug)),
            ),
            data=DataConfig(
                dir=_resolve(base, data_raw.get("dir", defaults.data.dir)),
            ),
            vector_store=VectorStoreConfig(
                collection=str(store_raw.get("collection", defaults.vector_store.collection)),
                persist_directory=_resolve(
                    base,
                    store_raw.get("persist_directory", defaults.vector_store.persist_directory),
                ),
                distance_metric=str(
                    store_raw.get("distance_metric", defaults.vector_store.distance_metric)
                ).lower(),
            ),
            embedding=EmbeddingConfig(
                model=str(embed_raw.get("model", defaults.embedding.model)),
                use_ollama=bool(embed_raw.get("use_ollama", defaults.embedding.use_ollama)),
                batch_size=int(embed_raw.get("batch_size", defaults.embedding.batch_size)),
                max_retries=int(embed_raw.get("max_retries", defaults.embedding.max_retries)),
                retry_delay=float(embed_raw.get("retry_delay", defaults.embedding.retry_delay)),
                batch_delay=float(embed_raw.get("batch_delay", defaults.embedding.batch_delay)),
            ),
            chunking=ChunkingConfig(
                chunk_size=int(chunk_raw.get("chunk_size", defaults.chunking.chunk_size)),
                chunk_overlap=int(
                    chunk_raw.get("chunk_overlap", defaults.chunking.chunk_overlap)
                ),
            ),
            llm=LLMConfig(
                backend=str(llm_raw.get("backend", defaults.llm.backend)).lower(),
                temperature=float(llm_raw.get("temperature", defaults.llm.temperature)),
                max_tokens=int(llm_raw.get("max_tokens", defaults.llm.max_tokens)),
                ollama_model=str(ollama_raw.get("model", defaults.llm.ollama_model)),
                groq_model=str(groq_raw.get("model", defaults.llm.groq_model)),
                api_key_env=str(groq_raw.get("api_key_env", defaults.llm.api_key_env)),
            ),
            retrieval=RetrievalConfig(
                top_k=int(retrieval_raw.get("top_k", defaults.retrieval.top_k)),
                min_score=float(retrieval_raw.get("min_score", defaults.retrieval.min_score)),
                summarize=bool(retrieval_raw.get("summarize", defaults.retrieval.summarize)),
            ),
            ingestion=IngestionConfig(
                ingest_on_startup=bool(
                    ingest_raw.get("ingest_on_startup", defaults.ingestion.ingest_on_startup)
                ),
                default_mode=str(
                    ingest_raw.get("default_mode", defaults.ingestion.default_mode)
                ).lower(),
                max_upload_mb=int(
                    ingest_raw.get("max_upload_mb", defaults.ingestion.max_upload_mb)
                ),
                upload_subdir=str(
                    ingest_raw.get("upload_subdir", defaults.ingestion.upload_subdir)
                ),
            ),
            source_path=path,
        )
    except (TypeError, ValueError) as ex:
        raise ConfigError(f"Bad value in {path}: {ex}") from ex

    _validate(cfg)
    return cfg


def describe(cfg: "Config") -> str:
    """Short human-readable summary, printed at startup."""
    return (
        f"config      : {cfg.source_path}\n"
        f"data dir    : {cfg.data.dir}\n"
        f"vector store: {cfg.vector_store.persist_directory} "
        f"(collection '{cfg.vector_store.collection}')\n"
        f"embedding   : {cfg.embedding.model} "
        f"({'Ollama' if cfg.embedding.use_ollama else 'SentenceTransformer'})\n"
        f"llm         : {cfg.llm.model} via {cfg.llm.backend} "
        f"(temp {cfg.llm.temperature}, max_tokens {cfg.llm.max_tokens})\n"
        f"chunking    : {cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap}\n"
        f"retrieval   : top_k {cfg.retrieval.top_k}, min_score {cfg.retrieval.min_score}"
    )


# Loaded once at import. Import the module rather than the object if you need
# to reload it (tests do this via load_config directly).
CONFIG = load_config()
