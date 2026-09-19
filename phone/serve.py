"""Serves phone/index.html to the phone over HTTPS, and makes the certificate the relay also needs.

    python phone/serve.py                 # prints the https:// URL to open on the phone

Why HTTPS for a page on your own LAN: browsers only hand out the camera in a
"secure context". That means HTTPS, or an origin of localhost. The phone reaches
this laptop at something like 192.168.1.20, which is neither, so a plain
http:// page gets no camera on iOS Safari or Android Chrome — the request fails
with NotAllowedError and no prompt is ever shown.

So this script:
  1. Generates a self-signed certificate valid for this laptop's LAN IP
     (cert.pem / key.pem, gitignored) if one is not already there.
  2. Serves phone/index.html over HTTPS using it.

The same cert.pem / key.pem pair is what you pass to the relay:

    python memory_pipeline.py --source wss://0.0.0.0:8765 --cert phone/cert.pem --key phone/key.pem ...

Both must be HTTPS/WSS and both must use the SAME certificate, so the phone only
has to be told to trust one thing.

First connection from the phone, once per laptop:
  - Open the printed https:// URL. Safari warns the certificate is untrusted
    (expected: nobody signs a certificate for a private IP). Tap Show Details ->
    visit this website.
  - Then open https://<laptop-ip>:8765 once and accept the same warning, so the
    WebSocket to that port is allowed too. It will show an empty page or an
    error body; that is fine, the point is accepting the certificate.
"""
import argparse
import functools
import http.server
import socket
import ssl
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT, KEY = HERE / "cert.pem", HERE / "key.pem"


def lan_ip() -> str:
    """This laptop's address on the local network, as the phone will reach it.

    Opens a UDP socket toward a public address and reads back which local
    interface the OS picked. No packet is actually sent, and it does not need
    the internet to work — it only asks the routing table a question.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip: str) -> None:
    """Create a self-signed cert covering this IP, unless one already exists.

    The IP goes in a subjectAltName, not just the common name: browsers have
    ignored the common name for host matching for years and will reject a cert
    without a matching SAN even after you accept the warning.
    """
    if CERT.exists() and KEY.exists():
        return
    print(f"generating a self-signed certificate for {ip} ...")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(KEY), "-out", str(CERT), "-days", "365",
         "-subj", "/CN=compass-laptop",
         "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
        check=True, capture_output=True,
    )
    print(f"wrote {CERT.name} and {KEY.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--relay-port", type=int, default=8765, help="only used for the printed instructions")
    args = ap.parse_args()

    ip = lan_ip()
    try:
        ensure_cert(ip)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"could not generate a certificate with openssl: {e}\n"
                 f"Install openssl, or drop your own cert.pem/key.pem into {HERE}.")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)

    # Serve only this directory, whatever the shell's working directory is.
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"\n  On the phone, open:  https://{ip}:{args.port}/")
    print(f"  Accept the certificate warning, then also visit https://{ip}:{args.relay_port} once")
    print(f"  and accept it there, so the WebSocket to the relay is allowed.\n")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
