"""Transport layer of pi's client package: the byte-transport contract.

The unix transport is a separate import (`pidrei_client.unix`), matching pi's
`@earendil-works/pi-client/unix` subpath export. The protocol client itself is
not ported — UPSTREAM_EXPERIMENTAL_RULING.md.
"""

from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers


__all__ = [
    "ByteTransport",
    "ByteTransportFactory",
    "ByteTransportHandlers",
]
