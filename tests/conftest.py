"""Shared test fixtures."""

import uuid

import psycopg
import pytest

from app import db


@pytest.fixture
def conn():
    """A database connection inside a throwaway schema; skips if Postgres isn't running."""
    try:
        connection = db.connect()
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (docker compose up -d db)")

    schema = f"test_{uuid.uuid4().hex[:8]}"
    connection.execute(f"CREATE SCHEMA {schema}")
    connection.execute(f"SET search_path TO {schema}, public")  # public holds the vector extension
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.execute(f"DROP SCHEMA {schema} CASCADE")
    connection.commit()
    connection.close()
