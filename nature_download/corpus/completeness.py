"""Fail-closed on-disk completeness checks for harvested article directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FIGURE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class ArticlePayloadStatus:
    """Deterministic inventory of the assets required to retain an article."""

    article_dir: Path
    figures: tuple[Path, ...]
    source_data: tuple[Path, ...]
    metadata: tuple[Path, ...]
    reasons: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.reasons


class IncompleteArticlePayloadError(ValueError):
    """Raised when an article lacks a figure, source-data file, or metadata."""


def _is_regular_nonempty_file(path: Path) -> bool:
    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and not path.name.startswith(".")
            and path.stat().st_size > 0
        )
    except OSError:
        return False


def _regular_nonempty_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if _is_regular_nonempty_file(path)
            ),
            key=lambda path: path.name.casefold(),
        )
    )


def inspect_article_payload(article_dir: str | Path) -> ArticlePayloadStatus:
    """Return whether an article has image, per-figure source data, and metadata.

    The check deliberately uses only the three standardized article subdirectories,
    so it is stable across the Nature, eLife, EMBO, and LSA harvesters.
    """

    article = Path(article_dir)
    if not article.is_dir() or article.is_symlink():
        return ArticlePayloadStatus(article, (), (), (), ("article-not-directory",))

    figures = tuple(
        path
        for path in _regular_nonempty_files(article / "figures")
        if path.suffix.casefold() in FIGURE_SUFFIXES
    )
    source_data = _regular_nonempty_files(article / "source_data")
    metadata = tuple(
        path
        for path in _regular_nonempty_files(article / "meta")
        if path.suffix.casefold() == ".json"
    )
    reasons: list[str] = []
    if not figures:
        reasons.append("figure-image-missing")
    if not source_data:
        reasons.append("source-data-missing")
    if not metadata:
        reasons.append("metadata-missing")
    return ArticlePayloadStatus(
        article,
        figures,
        source_data,
        metadata,
        tuple(reasons),
    )


def require_complete_article_payload(article_dir: str | Path) -> ArticlePayloadStatus:
    """Fail before publishing a staging directory that lacks required assets."""

    status = inspect_article_payload(article_dir)
    if not status.complete:
        raise IncompleteArticlePayloadError(
            f"incomplete article payload {status.article_dir}: "
            + ",".join(status.reasons)
        )
    return status


def quarantine_article_path(
    article_path: str | Path,
    *,
    bucket: str = "_rejected_no_source",
) -> Path | None:
    """Move an article path aside without deleting it, avoiding name collisions."""

    article = Path(article_path)
    if not article.exists() and not article.is_symlink():
        return None
    if not bucket.startswith("_"):
        raise ValueError("quarantine bucket must begin with '_'")
    destination_root = article.parent / bucket
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / article.name
    suffix = 2
    while destination.exists() or destination.is_symlink():
        destination = destination_root / f"{article.name}-{suffix}"
        suffix += 1
    article.replace(destination)
    return destination


def quarantine_incomplete_article(article_dir: str | Path) -> Path | None:
    """Quarantine an incomplete article directory; complete directories are kept."""

    article = Path(article_dir)
    if inspect_article_payload(article).complete:
        return None
    return quarantine_article_path(article)


def quarantine_incomplete_article_dirs(content_root: str | Path) -> list[Path]:
    """Quarantine payload-like direct children that fail the completeness gate."""

    root = Path(content_root)
    if not root.is_dir():
        return []
    quarantined: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        if not (
            (child / "figures").exists()
            or (child / "source_data").exists()
            or (child / "meta").exists()
        ):
            continue
        destination = quarantine_incomplete_article(child)
        if destination is not None:
            quarantined.append(destination)
    return quarantined
