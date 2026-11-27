"""pidrei-only: `describe_transport_error` reports a failed exchange from the
class of the failure and the host of its request (pi's SDKs say
`Connection error.` for all of them)."""

import pytest

from pidrei_http import http


def _request() -> http.Request:
    return http.Request("POST", "https://api.example.com/v1/messages")


# The transport's own text is what a punkreq error carries; none of it may reach the report.
_TRANSPORT_TEXT = "Failed to connect to https://api.example.com:443: [Errno 101] Network is unreachable"


@pytest.mark.parametrize(
    ("error_class", "expected"),
    [
        # The phase timeouts are `TimeoutException`s: each keeps its own report.
        (http.ConnectTimeout, "Timed out connecting to api.example.com"),
        (http.PoolTimeout, "Timed out waiting for a free connection to api.example.com"),
        (http.ReadTimeout, "Timed out waiting for a response from api.example.com"),
        (http.RequestTimeout, "Connection to api.example.com timed out"),
        (http.ConnectError, "Could not connect to api.example.com"),
        (http.ProxyError, "Could not reach api.example.com through the proxy"),
        (http.ReadError, "Connection to api.example.com was lost"),
        (http.WriteError, "Connection to api.example.com was lost"),
        (http.RemoteProtocolError, "Connection to api.example.com was lost"),
        (http.TransportError, "Network error while talking to api.example.com"),
    ],
)
def test_reports_a_failure_from_its_class_and_host(error_class, expected):
    assert http.describe_transport_error(error_class(_TRANSPORT_TEXT, request=_request())) == expected


def test_names_no_host_for_an_error_raised_without_a_request():
    assert http.describe_transport_error(http.ConnectError(_TRANSPORT_TEXT)) == "Could not connect to the server"
