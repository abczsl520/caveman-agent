"""Allow running as: python -m caveman.mcp.server"""
if __name__ == "__main__":
    from caveman.mcp.server import run_stdio
    run_stdio()
