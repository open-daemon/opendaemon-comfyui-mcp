"""Entry points. The modules import each other by plain name, so their folder goes first on the path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    """The MCP server (stdio). Starts the video pipeline web server by itself when needed."""
    import mcp_server
    mcp_server.main()


def web():
    """Only the web page and pipeline server, on http://127.0.0.1:8190 (PIPELINE_PORT)."""
    import server
    server.main()


if __name__ == "__main__":
    main()
