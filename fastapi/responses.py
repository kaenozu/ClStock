class HTMLResponse:
    """簡易的なHTMLレスポンス表現"""

    def __init__(self, content: str = "", status_code: int = 200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/html"}

    def __iter__(self):
        yield from ()
