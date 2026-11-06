"""Cookie-jar helpers that need nothing but the standard library.

Challenge handlers have to write cookies into a session's jar without importing
curl_cffi: the daemon proxy and the challenge registry both run on lightweight
paths, and ``curl_cffi.requests.cookies.Cookies`` would drag the heavy package
in.  The stdlib :class:`http.cookiejar.CookieJar` is exactly what curl_cffi
uses underneath, so a plain :class:`http.cookiejar.Cookie` is enough.

This module is stdlib-only on purpose.
"""

from __future__ import annotations

from http.cookiejar import Cookie, CookieJar

__all__ = ["seed_cookie_jar", "set_cookie"]


def set_cookie(
    jar: CookieJar,
    name: str,
    value: str,
    *,
    domain: str = ".zhihu.com",
    path: str = "/",
) -> None:
    """Store one cookie in *jar*, mirroring ``Cookies(jar).set(...)``.

    curl_cffi's ``Cookies.set`` builds a stdlib :class:`http.cookiejar.Cookie`
    with these exact fields; replicating it keeps the resulting jar byte-for-byte
    compatible with a cookie libcurl wrote itself.

    :param jar: The jar to mutate.
    :param name: Cookie name.
    :param value: Cookie value.
    :param domain: Cookie domain; the leading dot makes it a domain cookie.
    :param path: Cookie path.
    """
    cookie = Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=bool(domain),
        domain_initial_dot=domain.startswith("."),
        path=path,
        path_specified=bool(path),
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={"HttpOnly": None},
        rfc2109=False,
    )
    jar.set_cookie(cookie)


def seed_cookie_jar(jar: CookieJar, cookie_header: str) -> None:
    """Seed *jar* from a profile's ``Cookie`` header.

    Necessary because libcurl replaces an explicit ``Cookie`` header with the
    jar's contents as soon as the jar is non-empty — an unseeded jar would drop
    the profile's authentication cookies.  Names already present are left
    untouched so values updated by the server (e.g. a refreshed ``d_c0`` after
    a CAPTCHA) win over the stale header.

    :param jar: The session's cookie jar.
    :param cookie_header: Raw ``Cookie`` header value, ``name=value; ...``.
    """
    if not cookie_header:
        return
    existing = {cookie.name for cookie in jar}
    for part in cookie_header.split("; "):
        name, sep, value = part.partition("=")
        if not sep or not name or name in existing:
            continue
        try:
            set_cookie(jar, name, value)
        except Exception:
            continue
        existing.add(name)
