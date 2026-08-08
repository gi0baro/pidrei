"""Validated, framed message codec (port of pi protocol `codec.ts`).

`_is_protocol_value` is pi's plain-value gate: it rejects cycles and non-plain
objects (class instances, `bytes` in JSON-valued fields) before the schema
check ever traverses them, and bounds every error message so a rejected
payload is never retained or echoed at full size.
"""

from jsonschema import Draft202012Validator

from .cbor import CborOptions, decode_cbor, encode_cbor
from .framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FrameDecoder,
    FrameDecoderOptions,
    assert_complete_frame,
    encode_frame,
)
from .schemas import CLIENT_MESSAGE_SCHEMA, PROTOCOL_VERSION, SERVER_MESSAGE_SCHEMA, ClientMessage, ServerMessage


class ProtocolValidationError(Exception):
    pass


_client_message_validator = Draft202012Validator(CLIENT_MESSAGE_SCHEMA)
_server_message_validator = Draft202012Validator(SERVER_MESSAGE_SCHEMA)


def _is_protocol_value(value: object, ancestors: set[int]) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if not isinstance(value, list | dict) or id(value) in ancestors:
        return False
    ancestors.add(id(value))
    try:
        if isinstance(value, list):
            return all(_is_protocol_value(item, ancestors) for item in value)
        if type(value) is not dict:
            return False
        return all(isinstance(key, str) and _is_protocol_value(item, ancestors) for key, item in value.items())
    finally:
        ancestors.discard(id(value))


def parse_client_message(value: object) -> ClientMessage:
    if not _is_protocol_value(value, set()) or not _client_message_validator.is_valid(value):
        raise ProtocolValidationError("Invalid client protocol message")
    return value


def parse_server_message(value: object) -> ServerMessage:
    if not _is_protocol_value(value, set()) or not _server_message_validator.is_valid(value):
        raise ProtocolValidationError("Invalid server protocol message")
    return value


def _bounded_error_message(error: object) -> str:
    if not isinstance(error, Exception):
        return "Unknown codec error"
    message = str(error)
    return message if len(message) <= 500 else f"{message[:497]}..."


def _max_frame_length(options: FrameDecoderOptions | None) -> int:
    if options is not None and options.max_frame_length is not None:
        return options.max_frame_length
    return DEFAULT_MAX_FRAME_LENGTH


def _encode_protocol_message(value, parse, kind: str, options: FrameDecoderOptions | None) -> bytes:
    validated = parse(value)
    try:
        max_frame_length = _max_frame_length(options)
        frame = encode_frame(encode_cbor(validated, CborOptions(max_byte_length=max_frame_length)))
        assert_complete_frame(frame, FrameDecoderOptions(max_frame_length=max_frame_length))
        return frame
    except ProtocolValidationError:
        raise
    except Exception as error:
        raise ProtocolValidationError(f"Unable to encode {kind} protocol message: {_bounded_error_message(error)}")


def encode_client_message(message: ClientMessage, options: FrameDecoderOptions | None = None) -> bytes:
    """Validates and encodes one complete length-prefixed client message."""
    return _encode_protocol_message(message, parse_client_message, "client", options)


def encode_server_message(message: ServerMessage, options: FrameDecoderOptions | None = None) -> bytes:
    """Validates and encodes one complete length-prefixed server message."""
    return _encode_protocol_message(message, parse_server_message, "server", options)


class _ValidatedMessageDecoder:
    def __init__(self, kind: str, parse, options: FrameDecoderOptions | None):
        self._failed = False
        self._frames = FrameDecoder(options)
        self._kind = kind
        self._max_frame_length = _max_frame_length(options)
        self._parse = parse

    def push(self, chunk: bytes) -> list:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            messages = []
            for frame in self._frames.push(chunk):
                messages.append(self._parse(decode_cbor(frame, CborOptions(max_byte_length=self._max_frame_length))))
            return messages
        except Exception as error:
            self._failed = True
            if isinstance(error, ProtocolValidationError):
                raise
            raise ProtocolValidationError(f"Invalid {self._kind} protocol frame: {_bounded_error_message(error)}")

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol framing: {_bounded_error_message(error)}")


class ClientMessageDecoder:
    """Incrementally decodes and validates framed client messages."""

    def __init__(self, options: FrameDecoderOptions | None = None):
        self._decoder = _ValidatedMessageDecoder("client", parse_client_message, options)

    def push(self, chunk: bytes) -> list[ClientMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    """Incrementally decodes and validates framed server messages."""

    def __init__(self, options: FrameDecoderOptions | None = None):
        self._decoder = _ValidatedMessageDecoder("server", parse_server_message, options)

    def push(self, chunk: bytes) -> list[ServerMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def create_client_message_decoder(options: FrameDecoderOptions | None = None) -> ClientMessageDecoder:
    return ClientMessageDecoder(options)


def create_server_message_decoder(options: FrameDecoderOptions | None = None) -> ServerMessageDecoder:
    return ServerMessageDecoder(options)


def is_supported_protocol_version(version: object) -> bool:
    return isinstance(version, int | float) and not isinstance(version, bool) and version == PROTOCOL_VERSION
