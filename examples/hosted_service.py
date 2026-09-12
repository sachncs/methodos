"""Start the methodos FastAPI service via uvicorn.

Usage:
    python examples/hosted_service.py
    # or with custom host/port:
    HOST=0.0.0.0 PORT=8080 python examples/hosted_service.py

Requires: `pip install methodos`
"""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "methodos.service:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
