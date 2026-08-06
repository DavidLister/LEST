import sqlite3

from typer.testing import CliRunner

from lest.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "status"):
        assert command in result.output


def test_sqlite_vec_loads():
    """Canary: the interpreter must support loadable extensions and ship sqlite-vec."""
    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    (version,) = conn.execute("SELECT vec_version()").fetchone()
    assert version.startswith("v")
    conn.close()
