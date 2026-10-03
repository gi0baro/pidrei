"""pidrei's HTTP seam, shared by every package that makes HTTP requests: the
pooled clients and timeouts (`http`), proxy-env resolution (`http_proxy`),
PKCE (`pkce`) and the loopback callback server OAuth flows redirect to
(`callback_server`). The only pidrei package that imports punkreq or httpunk;
depends on no other pidrei package.
"""
