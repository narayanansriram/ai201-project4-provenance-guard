# In-memory store mapping content_id → record dict.
# Records survive the request lifecycle but not a server restart.

_store: dict = {}


def save(content_id: str, record: dict) -> None:
    _store[content_id] = record


def get(content_id: str) -> dict | None:
    return _store.get(content_id)


def update_status(content_id: str, status: str) -> bool:
    """Returns False if content_id not found."""
    if content_id not in _store:
        return False
    _store[content_id]["status"] = status
    return True
