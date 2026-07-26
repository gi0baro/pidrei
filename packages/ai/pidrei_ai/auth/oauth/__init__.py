"""OAuth login flows (port of pi's packages/ai/src/auth/oauth/).

Nothing here is imported by a provider factory: providers advertise OAuth
through `lazy_oauth`, and the flows load on first use via `load.py`. pi does
this so bundlers cannot follow the import into Node-only code; pidrei keeps it
so a plain `pidrei` start never pays for the flows, the callback servers, or
their sockets.

`radius.ts` is deliberately absent: the radius provider is a pi-specific
service, dropped in Phase 7 step 1 along with the server presence integration.
"""
