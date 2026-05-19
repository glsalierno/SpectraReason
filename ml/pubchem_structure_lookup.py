"""
Resolve **SMILES** and **InChI** via **PubChem PUG REST** when spectra metadata lacks structure.

This complements local JCAMP fields and mirrors the workflow of PubChem CAS/name fetch tools
(e.g. `cas-to-chem-data`, `pubchem-toxinfo-cas-retriever`) but stays **stdlib-only** (``urllib``)
so it drops into the FTIR repo without extra deps.

Typical resolution order (PubChem PUG REST):

1. Normalized **CAS** from metadata (``cas`` / ``CAS`` / ``CAS REGISTRY NO``) or
   parsed from path/filename when present.
2. If CAS yields no structure: **compound name** from metadata (``title`` /
   ``name`` / ``TITLE`` / ``NAME``) or cleaned filename stem.

Results are merged into metadata as ``SMILES``, ``INCHI``, ``INCHIKEY``, plus
``pubchem_*`` provenance. ``ml.structural_fg_svm`` then **canonicalizes SMILES
with RDKit** before Mordred.

PubChem docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest  
Related tooling: [cas-to-chem-data](https://github.com/glsalierno/cas-to-chem-data),
[pubchem-toxinfo-cas-retriever](https://github.com/glsalierno/pubchem-toxinfo-cas-retriever).

Networking (``urllib``):

- **Retries:** ``_http_get_json`` retries transient failures including ``http.client.IncompleteRead``
  (truncated chunked responses) and ``RemoteDisconnected``; each request uses ``Connection: close``.
  On those errors the module **drops the cached urllib opener** so the next attempt uses a fresh TLS
  stack. Override attempts with env ``PUBCHEM_HTTP_MAX_ATTEMPTS`` (integer ≥ 1).
- **Proxies:** ``HTTP_PROXY`` / ``HTTPS_PROXY`` / system proxy (via ``getproxies()``) are applied.
- **Contact:** set ``PUBCHEM_CONTACT_EMAIL`` for a clearer ``User-Agent`` string (NCBI guidance).
- **SSL:** prefers the ``truststore`` package (uses the **OS trust store**, best on Windows). Else ``certifi``.
- **Last resort:** ``PUBCHEM_INSECURE_SSL=1`` disables TLS verification (insecure; only for broken corporate PKI).
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_SCHEMA = 2
# Bump when normalization/ranking changes; stale caches retry network.
LOOKUP_STRATEGY = "cas_name_path_ranked_v4"

CAS_FIELD_KEYS = ("cas", "CAS", "CAS REGISTRY NO", "cas_registry_no", "rn", "registry_number")
NAME_FIELD_KEYS = ("title", "TITLE", "name", "NAME", "compound_name", "chemical_name")
FORMULA_FIELD_KEYS = ("formula", "FORMULA", "molecular_formula")
INCHI_FIELD_KEYS = ("inchi", "INCHI", "InChI")
INCHIKEY_FIELD_KEYS = ("inchikey", "INCHIKEY", "InChIKey")


def _cids_from_pug_json(data: dict[str, Any]) -> list[int]:
    """Parse CID list from PUG REST ``.../cids/JSON`` (``IdentifierList`` or legacy ``InformationList``)."""
    if not data:
        return []
    ident = data.get("IdentifierList") or {}
    raw = ident.get("CID")
    if raw is not None:
        return [int(x) for x in raw if x is not None]
    infos = data.get("InformationList", {}).get("Information", [])
    if not infos:
        return []
    raw2 = infos[0].get("CID")
    if raw2 is None:
        return []
    if isinstance(raw2, list):
        return [int(x) for x in raw2 if x is not None]
    return [int(raw2)]

# CAS registry number pattern (flexible leading segment).
_CAS_RE = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")
_NOISE_NAME_TOKEN_RE = re.compile(
    r"\b("
    r"ftir|ir|raman|nmr|mass\s+spectrum|spectrum|spectra|library|nist|sample|vendor|instrument|"
    r"kbr|nujol|film|liquid|gas|vapor|solid|neat|atr|transmittance|absorbance|"
    r"background|reference|std|standard"
    r")\b",
    flags=re.I,
)
# Remove parentheticals that contain only spectroscopic / sampling noise (not substituents).
_NOISE_ONLY_PAREN_RE = re.compile(
    r"\(\s*[^)]*\b(?:"
    r"IR|FTIR|RAMAN|NMR|MASS\s+SPECTRUM|SPECTRUM|KBR|Nujol|film|liquid|gas|vapor|solid|neat|"
    r"ATR|transmittance|absorbance|reference|standard|sample"
    r")\b[^)]*\)",
    flags=re.I,
)
_MIXTURE_HINT_RE = re.compile(
    r"\b(mixture|solution|copolymer|resin|oil|extract|standard\s*mix)\b",
    flags=re.I,
)
# Only strip parentheticals that look like sampling / physical state (NIST-style),
# not chemical substituents like "(O-CHLOROPHENYL)" or "(5-chloro-...)".
_SAMPLING_PAREN_RE = re.compile(
    r"\(\s*[^)]{0,120}?(?:"
    r"nujol|mull|kbr|cscl|pellet|film|solid|liquid|gas|powder|"
    r"solution|slurry|matrix|dispersion|transmission|reflectance|"
    r"wt\.?\s*%|vol\.?\s*%|\d{1,3}\s*%|percent|approx|tech\.?|technical|grade"
    r")[^)]{0,80}?\)",
    flags=re.I,
)

# Legacy valence / mineral naming → modern element oxidation synonyms (PubChem-oriented).
VALENCE_ALIASES: dict[str, list[str]] = {
    "ferrous": ["iron(II)", "iron(2+)", "iron"],
    "ferric": ["iron(III)", "iron(3+)", "iron"],
    "cuprous": ["copper(I)", "copper(1+)", "copper"],
    "cupric": ["copper(II)", "copper(2+)", "copper"],
    "manganous": ["manganese(II)", "manganese(2+)", "manganese"],
    "manganic": ["manganese(III)", "manganese(3+)", "manganese"],
    "stannous": ["tin(II)", "tin(2+)", "tin"],
    "stannic": ["tin(IV)", "tin(4+)", "tin"],
    "chromous": ["chromium(II)", "chromium(2+)", "chromium"],
    "chromic": ["chromium(III)", "chromium(3+)", "chromium"],
}

HYDRATE_ALIASES: dict[str, list[str]] = {
    "monohydrate": ["hydrate", "1-hydrate"],
    "hemihydrate": ["0.5-hydrate", "half-hydrate"],
    "dihydrate": ["hydrate", "2-hydrate"],
    "trihydrate": ["hydrate", "3-hydrate"],
    "pentahydrate": ["hydrate", "5-hydrate"],
}


def cas_checksum_is_valid(cas: str | None) -> bool:
    """Validate CAS checksum for ``NNNNNNN-NN-N`` identifiers."""
    cas_n = normalize_cas(cas)
    if not cas_n:
        return False
    left, check = cas_n.rsplit("-", 1)
    digits = left.replace("-", "")
    if not digits.isdigit() or not check.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits), start=1):
        total += i * int(ch)
    return (total % 10) == int(check)


def _extract_cas_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    found = _CAS_RE.findall(str(text))
    out: list[str] = []
    for raw in found:
        cas_n = normalize_cas(raw)
        if cas_n and cas_checksum_is_valid(cas_n):
            out.append(cas_n)
    return out


def _get_first_nonempty(md: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = md.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def normalize_chemical_name(raw: str | None, mode: str) -> str | None:
    """
    Chemical-name variants for PubChem queries.

    Parenthetical substituents (e.g. ``(O-CHLOROPHENYL)``) are preserved except in
    ``noise`` mode, where only spectroscopic/sample-only parentheses/tokens are stripped.
    Hyphens in names are kept (bonds/positions).
    """
    if not raw:
        return None
    s = str(raw).strip()
    if len(s) > 480:
        s = s[:480].rsplit(",", 1)[0].strip()
    if mode == "raw":
        s = re.sub(r"\s+", " ", s)
        return s if len(s) >= 2 else None
    if mode == "light":
        r0 = normalize_chemical_name(raw, "raw")
        if not r0:
            return None
        s = r0.replace("′", "'").replace("`", "'").replace("´", "'")
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s) >= 2 else None
    if mode == "noise":
        r0 = normalize_chemical_name(raw, "light")
        if not r0:
            return None
        s = r0
        s = _NOISE_ONLY_PAREN_RE.sub(" ", s)
        s = _SAMPLING_PAREN_RE.sub(" ", s)
        s = _CAS_RE.sub(" ", s)
        s = _NOISE_NAME_TOKEN_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip(" -–—\t")
        return s if len(s) >= 2 else None
    return None


def _usable_name_query(name: str | None) -> bool:
    if not name:
        return False
    s = str(name).strip()
    if len(s) < 2:
        return False
    if _MIXTURE_HINT_RE.search(s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    return letters >= 2


def _dedupe_query_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in items:
        key = item["query"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_for_name_similarity(s: str | None) -> str:
    """Lowercase, collapse punctuation/separators for fuzzy synonym / overlap checks."""
    if not s:
        return ""
    t = str(s).lower()
    t = re.sub(r"[\s\-_',;:./\\]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def light_punctuation_normalized(raw: str | None) -> str | None:
    """Prime marks → ASCII apostrophe; whitespace collapsed (same as ``normalize_chemical_name(..., 'light')``)."""
    return normalize_chemical_name(raw, "light")


def uninvert_registry_name(name: str) -> str | None:
    """
    Conservative uninversion for NIST/registry comma names (not full IUPAC parsing).

    Examples:
        ``BENZOIC ACID, 4-CHLORO-`` → ``4-chlorobenzoic acid``
        ``ACETIC ACID, TRICHLORO-`` → ``trichloroacetic acid``
        ``ETHANOL, 2,2'-IMINODI-`` → ``2,2'-iminodiethanol``
    """
    s = str(name).strip()
    if not s:
        return None
    su = " ".join(s.split())

    # BENZOIC ACID, 4-CHLORO-  (substituent before acid name)
    if re.match(r"^BENZOIC\s+ACID\s*,", su, re.I):
        tail = su.split(",", 1)[1].strip().rstrip("-").strip()
        if tail and re.search(
            r"(?:\d|CHLORO|BROMO|FLUORO|IODO|METHYL|NITRO|AMINO|HYDROXY|ETHYL|PROP)",
            tail,
            re.I,
        ):
            tl = tail.lower()
            if not tl.endswith("acid"):
                return f"{tl}benzoic acid"

    # ACETIC ACID, TRICHLORO-
    if re.match(r"^ACETIC\s+ACID\s*,\s*TRICHLORO", su, re.I):
        return "trichloroacetic acid"
    if re.match(r"^ACETIC\s+ACID\s*,\s*DICHLORO", su, re.I):
        return "dichloroacetic acid"

    # ETHANOL, 2,2'-IMINODI-
    if re.match(r"^ETHANOL\s*,\s*2,2['\u2032\u2019]\s*-\s*IMINODI", su, re.I):
        return "2,2'-iminodiethanol"
    if re.match(r"^ETHANOL\s*,\s*2,2['\u2032\u2019]\s*IMINODI", su, re.I):
        return "2,2'-iminodiethanol"

    # ETHANOL, … DI  — trailing `` DI`` → ``…diethanol`` (registry style)
    m = re.match(r"^ETHANOL\s*,\s*(.+)\s+DI\s*$", su, re.I)
    if m:
        mid = m.group(1).strip()
        if "'" in mid or "\u2032" in mid or "\u2019" in mid or re.search(r"\d\s*,\s*\d", mid):
            return f"{mid}diethanol"

    return None


def expand_valence_aliases(name: str) -> list[str]:
    """Replace legacy valence words (ferrous, cupric, …) with chemically plausible synonyms."""
    out: list[str] = []
    lower = name.lower()
    for legacy, aliases in VALENCE_ALIASES.items():
        if not re.search(r"\b" + re.escape(legacy) + r"\b", lower):
            continue
        for alt in aliases:
            replaced = re.sub(r"\b" + re.escape(legacy) + r"\b", alt, name, flags=re.I)
            if replaced.strip() and replaced.strip().lower() != name.strip().lower():
                out.append(replaced.strip())
    return out


def normalize_salt_hydrate_terms(name: str) -> list[str]:
    """Alternate hydrate spellings / stoichiometry hints for lookup."""
    out: list[str] = []
    nl = name.lower()
    for word, aliases in HYDRATE_ALIASES.items():
        if word not in nl:
            continue
        for alt in aliases:
            rep = re.sub(re.escape(word), alt, name, flags=re.I)
            if rep.strip() and rep.strip().lower() != name.strip().lower():
                out.append(rep.strip())
    return out


def comma_reordered_name(name: str) -> str | None:
    """Swap ``head, tail`` — low-risk only for short single-comma strings."""
    if name.count(",") != 1:
        return None
    if len(name) > 120:
        return None
    a, b = name.split(",", 1)
    a, b = a.strip(), b.strip()
    if len(a) < 4 or len(b) < 4:
        return None
    return f"{b}, {a}"


def typo_spelling_variants(name: str) -> list[str]:
    """Very conservative US/UK spelling toggles (sulf… ↔ sulph…)."""
    out: list[str] = []
    if re.search(r"\bsulph", name, re.I):
        out.append(re.sub(r"\bsulph", "sulf", name, flags=re.I))
    elif re.search(r"\bsulf", name, re.I):
        out.append(re.sub(r"\bsulf", "sulph", name, count=1, flags=re.I))
    return out


def generate_name_variants(raw_name: str, source_key: str) -> list[dict[str, str]]:
    """
    Ordered name-interpretation variants for PubChem name lookup (synonym-oriented).

    Order: raw → light punctuation → spectral-noise strip → uninversions → valence →
    hydrate/salt → comma swap → spelling. Duplicate queries (case-insensitive) are skipped.
    """
    seq: list[tuple[str, str]] = []
    r = normalize_chemical_name(raw_name, "raw")
    li = normalize_chemical_name(raw_name, "light")
    no = normalize_chemical_name(raw_name, "noise")

    for nm, lab in ((r, "raw"), (li, "light_punct"), (no, "noise_strip")):
        if nm:
            seq.append((nm, lab))

    bases_for_transform = [x for x in (no, li, r) if x]
    seen_u: set[str] = set()
    for base in bases_for_transform:
        u = uninvert_registry_name(base)
        if u and u.lower() not in seen_u:
            seen_u.add(u.lower())
            seq.append((u, "uninvert"))

    seen_v: set[str] = set()
    for base in bases_for_transform:
        for v in expand_valence_aliases(base):
            k = v.lower()
            if k not in seen_v:
                seen_v.add(k)
                seq.append((v, "valence"))

    seen_h: set[str] = set()
    for base in bases_for_transform:
        for v in normalize_salt_hydrate_terms(base):
            k = v.lower()
            if k not in seen_h:
                seen_h.add(k)
                seq.append((v, "hydrate_salt"))

    seen_c: set[str] = set()
    for base in bases_for_transform:
        cr = comma_reordered_name(base)
        if cr and cr.lower() not in seen_c:
            seen_c.add(cr.lower())
            seq.append((cr, "comma_swap"))

    seen_t: set[str] = set()
    for base in bases_for_transform:
        for v in typo_spelling_variants(base):
            k = v.lower()
            if k not in seen_t:
                seen_t.add(k)
                seq.append((v, "typo"))

    out: list[dict[str, str]] = []
    for q, variant in seq:
        if q and _usable_name_query(q):
            out.append({"query": q.strip(), "variant": variant, "source_key": source_key})
    return out


def extract_cas_candidates(md: dict[str, Any], path_hint: str | None) -> tuple[list[dict[str, str]], int]:
    """Return ``(candidates, invalid_cas_token_count)`` (checksum failures from CAS-like strings)."""
    invalid = 0
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def try_add(raw: object | None, source: str) -> None:
        nonlocal invalid
        if raw is None:
            return
        s = str(raw).strip()
        if not s:
            return
        cas_n = normalize_cas(s)
        if not cas_n:
            return
        if not cas_checksum_is_valid(cas_n):
            if _CAS_RE.search(s):
                invalid += 1
            return
        if cas_n in seen:
            return
        seen.add(cas_n)
        out.append({"query": cas_n, "source": source})

    for k in CAS_FIELD_KEYS:
        try_add(md.get(k), f"meta:{k}")
    for nk in NAME_FIELD_KEYS:
        v = md.get(nk)
        if v:
            for tok in _extract_cas_tokens(str(v)):
                try_add(tok, f"meta:{nk}:token")
    if path_hint:
        for tok in _extract_cas_tokens(path_hint):
            try_add(tok, "path:token")
    return out, invalid


def extract_name_candidates(md: dict[str, Any], path_hint: str | None) -> list[dict[str, str]]:
    """
    Name-interpretation pipeline: raw/light/noise plus uninvert, valence, hydrate, comma swap, spelling.

    Lower priority: path stem variants after metadata fields.
    """
    name_candidates: list[dict[str, str]] = []

    def add(q: str | None, source: str) -> None:
        if not q or not _usable_name_query(q):
            return
        name_candidates.append({"query": str(q).strip(), "source": source})

    for key in NAME_FIELD_KEYS:
        val = md.get(key)
        if val is None or not str(val).strip():
            continue
        base = str(val).strip()
        for item in generate_name_variants(base, key):
            add(item["query"], f"meta:{key}:{item['variant']}")

    if path_hint:
        leaf = Path(str(path_hint).replace("\\", "/").split("/")[-1]).stem
        leaf = urllib.parse.unquote(leaf)
        for item in generate_name_variants(leaf, "path_leaf"):
            add(item["query"], f"path:stem:{item['variant']}")

    return _dedupe_query_items(name_candidates)


def extract_pubchem_query_bundle(md: dict[str, Any], path_hint: str | None) -> dict[str, Any]:
    """Dry-run bundle for debugging (no HTTP)."""
    cas_c, inv = extract_cas_candidates(md, path_hint)
    name_c = extract_name_candidates(md, path_hint)
    return {
        "cas_candidates": cas_c,
        "name_candidates": name_c,
        "invalid_cas_tokens": inv,
        "formula_raw": _get_first_nonempty(md, FORMULA_FIELD_KEYS),
        "inchi_raw": _get_first_nonempty(md, INCHI_FIELD_KEYS),
        "inchikey_raw": _get_first_nonempty(md, INCHIKEY_FIELD_KEYS),
    }


def _build_query_candidates(
    md: dict[str, Any], path_hint: str | None
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    cas_list, invalid_n = extract_cas_candidates(md, path_hint)
    name_list = extract_name_candidates(md, path_hint)
    return cas_list, name_list, invalid_n


def normalize_cas(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().replace(" ", "")
    m = _CAS_RE.search(s)
    if not m:
        return None
    return m.group(1)


def cas_and_name_from_path_hint(path_str: str | None) -> tuple[str | None, str | None]:
    """
    Extract (CAS, name_guess) from ``source_path`` or ZIP member path
    (e.g. ``...::IR/Biphenyl_92-52-4-IR.jdx``).
    """
    if not path_str or not str(path_str).strip():
        return None, None
    leaf = str(path_str).replace("\\", "/").split("/")[-1]
    stem = Path(leaf).stem if leaf else ""
    if not stem:
        return None, None
    cas_matches = _CAS_RE.findall(stem)
    cas = normalize_cas(cas_matches[-1]) if cas_matches else None
    name_guess = _CAS_RE.sub(" ", stem)
    name_guess = re.sub(r"\s*[_\-]+\s*IR\s*$", "", name_guess, flags=re.I)
    name_guess = name_guess.replace("_", " ").strip(" -–—\t")
    if len(name_guess) < 2:
        name_guess = ""
    # Drop trailing CAS-like fragments already captured
    name_guess = _CAS_RE.sub("", name_guess).strip(" -–—")
    return cas, name_guess if len(name_guess) >= 2 else None


def _pubchem_user_agent() -> str:
    contact = (os.environ.get("PUBCHEM_CONTACT_EMAIL") or "").strip()
    if contact:
        return f"FTIR-chunks/1.0 (mailto:{contact}; PubChem PUG REST; +https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)"
    return (
        "FTIR-chunks/1.0 (PubChem PUG REST; academic; "
        "set PUBCHEM_CONTACT_EMAIL for a traceable User-Agent per NCBI guidance)"
    )


def _build_url_opener() -> urllib.request.OpenerDirector:
    """Opener with system/env proxies and TLS context (certifi preferred; optional insecure override)."""
    handlers: list[Any] = []
    proxies = urllib.request.getproxies()
    if proxies:
        handlers.append(urllib.request.ProxyHandler(proxies))

    insecure = os.environ.get("PUBCHEM_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes")
    if insecure:
        ctx = ssl._create_unverified_context()
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    else:
        ctx: ssl.SSLContext | None = None
        try:
            import truststore  # type: ignore[import-not-found]

            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except Exception:
            try:
                import certifi  # type: ignore[import-not-found]

                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = None
        if ctx is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        else:
            handlers.append(urllib.request.HTTPSHandler())
    handlers.append(urllib.request.HTTPRedirectHandler())
    return urllib.request.build_opener(*handlers)


_opener: urllib.request.OpenerDirector | None = None


def _reset_url_opener() -> None:
    """Drop cached opener so the next request builds fresh TLS/HTTP handlers (mitigates bad pooled state)."""
    global _opener
    _opener = None


def _url_opener() -> urllib.request.OpenerDirector:
    global _opener
    if _opener is None:
        _opener = _build_url_opener()
    return _opener


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Collect ``exc`` and linked ``__cause__`` / ``__context__`` (dedup by id)."""
    out: list[BaseException] = []
    seen: set[int] = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(cur)
        c = getattr(cur, "__cause__", None)
        if c is not None:
            stack.append(c)
        cx = getattr(cur, "__context__", None)
        if cx is not None and cx is not c:
            stack.append(cx)
    return out


def _http_get_json(
    url: str,
    *,
    timeout: float = 60.0,
    max_attempts: int = 8,
    backoff_base_s: float = 1.25,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    GET JSON from PubChem PUG REST with retries.

    Returns ``(data, None)`` on success, or ``(None, err)`` with a short diagnostic string
    for the last failure (HTTP body snippet, SSL message, etc.).
    """
    env_ma = (os.environ.get("PUBCHEM_HTTP_MAX_ATTEMPTS") or "").strip()
    if env_ma.isdigit() and int(env_ma) >= 1:
        max_attempts = int(env_ma)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _pubchem_user_agent(),
            "Accept": "application/json",
            # Avoid pooled connections that sometimes truncate chunked bodies (IncompleteRead).
            "Connection": "close",
        },
    )
    last_err: str | None = None
    for attempt in range(max_attempts):
        opener = _url_opener()
        try:
            with closing(opener.open(req, timeout=timeout)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw), None
        except urllib.error.HTTPError as e:
            code = getattr(e, "code", None)
            body = ""
            try:
                body = (e.read() or b"").decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            last_err = f"HTTPError {code}: {e.reason!s}" + (f" body={body!r}" if body else "")
            if code in (429, 500, 502, 503, 504):
                pass
            else:
                return None, last_err
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason!s}" if getattr(e, "reason", None) else f"URLError: {e!s}"
        except (TimeoutError, OSError) as e:
            last_err = f"{type(e).__name__}: {e!s}"
        except http.client.IncompleteRead as e:
            # Chunked PUG REST bodies occasionally truncate mid-read (proxies, TLS, server hiccups).
            try:
                n = len(e.partial)  # type: ignore[arg-type]
            except Exception:
                n = -1
            last_err = f"IncompleteRead: partial_bytes={n} msg={e!s}"
            _reset_url_opener()
        except http.client.RemoteDisconnected as e:
            last_err = f"RemoteDisconnected: {e!s}"
            _reset_url_opener()
        except (json.JSONDecodeError, ValueError) as e:
            return None, f"JSON decode: {e!s}"
        except Exception as e:
            # Some builds wrap the chunked-parser failure; match IncompleteRead anywhere in the chain.
            handled = False
            for cur in _exception_chain(e):
                if isinstance(cur, http.client.IncompleteRead):
                    try:
                        n = len(cur.partial)  # type: ignore[arg-type]
                    except Exception:
                        n = -1
                    last_err = f"IncompleteRead: partial_bytes={n} msg={cur!s}"
                    handled = True
                    break
                if isinstance(cur, http.client.RemoteDisconnected):
                    last_err = f"RemoteDisconnected: {cur!s}"
                    handled = True
                    break
            if handled:
                _reset_url_opener()
            else:
                raise

        if attempt + 1 >= max_attempts:
            break
        delay = backoff_base_s * (2**attempt) + random.uniform(0.0, 0.35)
        time.sleep(delay)

    return None, last_err or "unknown_error"


def diagnose_pubchem_cas_xref(cas: str, *, delay_s: float = 0.0) -> dict[str, Any]:
    """Single CAS→CID xref call for debugging (same URL stack as ``fetch_cids_by_cas``)."""
    cas_n = normalize_cas(cas)
    if not cas_n:
        return {"ok": False, "error": "invalid_cas", "cas": cas}
    if delay_s > 0:
        time.sleep(delay_s)
    url = f"{PUG_BASE}/compound/xref/RN/{urllib.parse.quote(cas_n, safe='')}/cids/JSON"
    data, err = _http_get_json(url)
    out: dict[str, Any] = {"url": url, "cas": cas_n, "ok": data is not None}
    if err:
        out["http_error"] = err
    if data:
        cids = _cids_from_pug_json(data)
        out["n_cids"] = len(cids)
        out["top_cids"] = cids[:5]
    return out


def fetch_cids_by_cas(cas: str, *, delay_s: float = 0.0) -> tuple[list[int], str | None]:
    cas_n = normalize_cas(cas)
    if not cas_n:
        return [], None
    if delay_s > 0:
        time.sleep(delay_s)
    url = f"{PUG_BASE}/compound/xref/RN/{urllib.parse.quote(cas_n, safe='')}/cids/JSON"
    data, err = _http_get_json(url)
    if not data:
        return [], err
    return _cids_from_pug_json(data), None


def fetch_cids_by_name(name: str, *, delay_s: float = 0.0) -> tuple[list[int], str | None]:
    name = str(name).strip()
    if len(name) < 2:
        return [], None
    if delay_s > 0:
        time.sleep(delay_s)
    enc = urllib.parse.quote(name, safe="")
    url = f"{PUG_BASE}/compound/name/{enc}/cids/JSON"
    data, err = _http_get_json(url)
    if not data:
        return [], err
    return _cids_from_pug_json(data), None


def fetch_structure_by_cid(cid: int, *, delay_s: float = 0.0) -> tuple[dict[str, Any], str | None]:
    if delay_s > 0:
        time.sleep(delay_s)
    url = (
        f"{PUG_BASE}/compound/cid/{int(cid)}/property/"
        "CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES,InChI,InChIKey,IUPACName,MolecularFormula,MolecularWeight/JSON"
    )
    data, err = _http_get_json(url)
    if not data:
        return {}, err
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return {}, err
    return dict(props[0]), None


def fetch_synonyms_for_cids(
    cids: list[int], *, delay_s: float = 0.0
) -> tuple[dict[int, list[str]], str | None]:
    """PubChem ``/compound/cid/.../synonyms/JSON`` (batch CID list supported)."""
    if not cids:
        return {}, None
    if delay_s > 0:
        time.sleep(delay_s)
    enc = ",".join(str(int(c)) for c in cids)
    url = f"{PUG_BASE}/compound/cid/{enc}/synonyms/JSON"
    data, err = _http_get_json(url)
    if not data:
        return {}, err
    out: dict[int, list[str]] = {}
    for info in data.get("InformationList", {}).get("Information", []):
        cid = info.get("CID")
        syns = info.get("Synonym") or info.get("Synonyms")
        if cid is None:
            continue
        if isinstance(syns, str):
            syns = [syns]
        elif not isinstance(syns, list):
            syns = []
        out[int(cid)] = [str(x) for x in syns if x]
    return out, None


def fetch_cids_by_fastformula(formula: str, *, delay_s: float = 0.0) -> tuple[list[int], str | None]:
    """
    ``/compound/fastformula/{formula}/cids/JSON`` — optional fallback when name queries fail.

    Formula should be a Hill-system or conventional molecular formula string (spaces optional).
    """
    mf = _normalize_formula_compact(formula)
    if len(mf) < 2:
        return [], None
    if delay_s > 0:
        time.sleep(delay_s)
    enc = urllib.parse.quote(mf, safe="")
    url = f"{PUG_BASE}/compound/fastformula/{enc}/cids/JSON"
    data, err = _http_get_json(url)
    if not data:
        return [], err
    return _cids_from_pug_json(data), None


def _tokenize_name(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if len(t) >= 2}


def _jaccard_tokens(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = a | b
    return len(a & b) / float(len(u)) if u else 0.0


def score_against_pubchem_synonyms(
    query: str,
    props: dict[str, Any],
    synonyms: list[str],
    *,
    max_synonyms: int = 450,
) -> tuple[float, list[str]]:
    """
    Rank query text against PubChem record (IUPACName + synonym list).

    Returns ``(bonus_score_0_to_~0.5, reason_strings)``.
    """
    reasons: list[str] = []
    qn = normalize_for_name_similarity(query)
    iupac = str(props.get("IUPACName") or "")
    pool: list[str] = [iupac]
    pool.extend(synonyms[:max_synonyms])

    best = 0.0
    best_lab = ""
    q_tokens = _tokenize_name(query)

    for syn in pool:
        if not syn:
            continue
        sn = normalize_for_name_similarity(str(syn))
        if not sn:
            continue
        if qn == sn:
            best = max(best, 0.48)
            best_lab = "exact_norm"
            continue
        if qn in sn or sn in qn:
            best = max(best, 0.36)
            best_lab = "substring_norm"
            continue
        sim = _jaccard_tokens(q_tokens, _tokenize_name(syn))
        if sim > 0:
            cand = min(0.30, sim * 0.38)
            if cand > best:
                best = cand
                best_lab = f"token_jacc:{sim:.2f}"

    if best > 0:
        reasons.append(f"synonym_evidence:{best_lab}:{best:.3f}")
    return best, reasons


def _metadata_has_formula(md: dict[str, Any]) -> bool:
    return bool(_get_first_nonempty(md, FORMULA_FIELD_KEYS))


def _formula_conflicts_metadata(md: dict[str, Any], mol_formula: str | None) -> bool:
    mf = _get_first_nonempty(md, FORMULA_FIELD_KEYS)
    if not mf or not mol_formula:
        return False
    return _normalize_formula_compact(mf) != _normalize_formula_compact(str(mol_formula))


def _is_transport_pubchem_error(err: str | None) -> bool:
    if not err:
        return False
    u = err.upper()
    return any(
        x in u
        for x in (
            "CERTIFICATE_VERIFY_FAILED",
            "SSL:",
            "TIMEOUT",
            "TEMPORARY FAILURE",
            "UNREACHABLE",
            "CONNECTION REFUSED",
            "CONNECTION RESET",
            "NAME OR SERVICE NOT KNOWN",
            "GETADDRINFO",
        )
    )


def resolve_pubchem(
    *,
    cas: str | None = None,
    name: str | None = None,
    delay_s: float = 0.25,
) -> dict[str, Any] | None:
    """
    Return dict with CanonicalSMILES (and fallbacks), InChI, InChIKey, CID, and provenance fields.
    """
    cas_n = normalize_cas(cas)
    tried: list[str] = []

    def pick_smiles(props: dict[str, Any]) -> str | None:
        # PubChem PUG may return ``CanonicalSMILES`` / ``IsomericSMILES`` or newer ``SMILES`` / ``ConnectivitySMILES``.
        for k in ("CanonicalSMILES", "IsomericSMILES", "SMILES", "ConnectivitySMILES"):
            v = props.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return None

    if cas_n:
        tried.append(f"cas:{cas_n}")
        cids, _err = fetch_cids_by_cas(cas_n, delay_s=delay_s)
        if cids:
            cid = int(cids[0])
            props, _perr = fetch_structure_by_cid(cid, delay_s=delay_s)
            smi = pick_smiles(props)
            inchi = props.get("InChI")
            ikey = props.get("InChIKey")
            if smi or inchi:
                return {
                    "SMILES": smi,
                    "INCHI": str(inchi).strip() if inchi else None,
                    "INCHIKEY": str(ikey).strip() if ikey else None,
                    "pubchem_cid": cid,
                    "pubchem_iupac_name": props.get("IUPACName"),
                    "pubchem_via": "cas",
                    "pubchem_query": cas_n,
                }

    if name and str(name).strip():
        nm = str(name).strip()
        # Trim overly long titles occasionally seen in JCAMP
        if len(nm) > 220:
            nm = nm[:220].rsplit(",", 1)[0].strip()
        tried.append(f"name:{nm[:80]}")
        cids, _err = fetch_cids_by_name(nm, delay_s=delay_s)
        if cids:
            cid = int(cids[0])
            props, _perr = fetch_structure_by_cid(cid, delay_s=delay_s)
            smi = pick_smiles(props)
            inchi = props.get("InChI")
            ikey = props.get("InChIKey")
            if smi or inchi:
                return {
                    "SMILES": smi,
                    "INCHI": str(inchi).strip() if inchi else None,
                    "INCHIKEY": str(ikey).strip() if ikey else None,
                    "pubchem_cid": cid,
                    "pubchem_iupac_name": props.get("IUPACName"),
                    "pubchem_via": "name",
                    "pubchem_query": nm,
                }

    return None


def _normalize_formula_compact(s: str) -> str:
    return re.sub(r"\s+", "", str(s).upper())


def _formula_matches_metadata(md: dict[str, Any], mol_formula: str | None) -> bool:
    mf = _get_first_nonempty(md, FORMULA_FIELD_KEYS)
    if not mf or not mol_formula:
        return False
    return _normalize_formula_compact(mf) == _normalize_formula_compact(str(mol_formula))


def _norm_title_cmp(s: str) -> str:
    t = str(s).lower().strip()
    t = t.replace(";", " ").replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def _exact_or_contained_name_match(query: str, iupac: str) -> tuple[bool, bool]:
    qn = _norm_title_cmp(query)
    un = _norm_title_cmp(iupac)
    if not qn or not un:
        return False, False
    if qn == un:
        return True, False
    if qn in un or un in qn:
        return False, True
    return False, False


def _inorganic_synonym_bonus(query: str, iupac: str) -> tuple[float, str | None]:
    """Map legacy mineral/common names to PubChem IUPAC-style strings."""
    ql = query.lower()
    il = iupac.lower().replace(";", " ")
    # Iron oxalates
    if "ferrous" in ql and "oxalate" in ql and "iron" in il and "oxalate" in il:
        if "2+" in il or "(ii)" in il or "iron(2" in il:
            return 0.38, "synonym_ferrous_iron2"
    if "ferric" in ql and "oxalate" in ql and "iron" in il and "oxalate" in il:
        if "3+" in il or "(iii)" in il or "iron(3" in il:
            return 0.38, "synonym_ferric_iron3"
    # Manganese
    if "manganous" in ql and "oxalate" in ql and "manganese" in il and "oxalate" in il:
        if "2+" in il or "(ii)" in il:
            return 0.36, "synonym_manganous"
    # Zinc / Mg etc.: token overlap handles; dilithium dipotassium
    if "lithium" in ql and "oxalate" in ql and "lithium" in il and "oxalate" in il:
        if "dilithium" in il or "lithium" in il:
            return 0.22, "lithium_salt_equiv"
    if "potassium" in ql and "oxalate" in ql and "potassium" in il and "oxalate" in il:
        if "dipotassium" in il:
            return 0.22, "potassium_salt_equiv"
    if "sodium" in ql and "acetate" in ql and "sodium" in il and "acetate" in il:
        return 0.18, "sodium_salt_equiv"
    return 0.0, None


def _inchikey_conflict(md: dict[str, Any], cand_ikey: str | None) -> bool:
    mb = _get_first_nonempty(md, INCHIKEY_FIELD_KEYS)
    if not mb or not cand_ikey:
        return False
    mp = mb.strip().upper().split("-")
    pp = str(cand_ikey).strip().upper().split("-")
    if not mp or not pp:
        return False
    if mp[0] != pp[0]:
        return True
    if len(mp) >= 2 and len(pp) >= 2 and mp[1] != pp[1]:
        return True
    return False


def _inchikey_matches_metadata(md: dict[str, Any], cand_ikey: str | None) -> bool:
    mb = _get_first_nonempty(md, INCHIKEY_FIELD_KEYS)
    if not mb or not cand_ikey:
        return False
    mp = mb.strip().upper().split("-")
    pp = str(cand_ikey).strip().upper().split("-")
    if mp[0] != pp[0]:
        return False
    if len(mp) >= 2 and len(pp) >= 2:
        return mp[1] == pp[1]
    return True


def resolve_pubchem_phase1(
    *,
    md: dict[str, Any],
    path_hint: str | None,
    delay_s: float = 0.25,
    metrics: dict[str, int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    """
    Resolver with ranked CID scoring and conservative acceptance.

    Returns ``(resolved_dict_or_none, attempts, transport_network_failure)``.
    ``transport_network_failure`` is True when PubChem could not be reached (TLS/timeout/etc.)
    and no usable compound data was retrieved.
    """
    cas_candidates, name_candidates, invalid_n = _build_query_candidates(md, path_hint)
    if metrics is not None and invalid_n:
        metrics["invalid_cas"] = metrics.get("invalid_cas", 0) + invalid_n

    attempts: list[dict[str, Any]] = []
    cid_cap = 10
    got_pubchem_payload = False
    transport_fail_seen = False

    def bump(key: str, n: int = 1) -> None:
        if metrics is not None:
            metrics[key] = metrics.get(key, 0) + n

    def pick_smiles(props: dict[str, Any]) -> str | None:
        for k in ("IsomericSMILES", "CanonicalSMILES", "SMILES", "ConnectivitySMILES"):
            v = props.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return None

    def rank_cids(
        query: str,
        query_type: str,
        source: str,
        cids: list[int],
        *,
        rank_context: str = "normal",
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
        nonlocal got_pubchem_payload, transport_fail_seen
        ranked: list[dict[str, Any]] = []
        q_tokens = _tokenize_name(query)
        path_weak = source.startswith("path:")
        meta_name = query_type == "name" and source.startswith("meta:")
        capped = [int(x) for x in cids[:cid_cap]]
        single_cid = len(capped) == 1

        syn_map: dict[int, list[str]] = {}
        if capped:
            sm, serr = fetch_synonyms_for_cids(capped, delay_s=delay_s)
            if sm:
                syn_map = sm
            if serr and _is_transport_pubchem_error(serr):
                transport_fail_seen = True

        for cid in capped:
            props, perr = fetch_structure_by_cid(int(cid), delay_s=delay_s)
            if props:
                got_pubchem_payload = True
            if perr and _is_transport_pubchem_error(perr):
                transport_fail_seen = True
            if not props:
                continue
            smi = pick_smiles(props)
            inchi = props.get("InChI")
            ikey = props.get("InChIKey")
            if not smi and not inchi:
                continue
            syns = syn_map.get(int(cid), [])
            syn_score, syn_rs = score_against_pubchem_synonyms(query, props, syns)

            if _inchikey_conflict(md, str(ikey) if ikey else None):
                ranked.append(
                    {
                        "cid": int(cid),
                        "score": -1.0,
                        "reasons": ["rejected_inchikey_mismatch"],
                        "synonym_evidence": syn_score,
                        "SMILES": smi,
                        "INCHI": str(inchi).strip() if inchi else None,
                        "INCHIKEY": str(ikey).strip() if ikey else None,
                        "IUPACName": props.get("IUPACName"),
                        "MolecularFormula": props.get("MolecularFormula"),
                        "MolecularWeight": props.get("MolecularWeight"),
                    }
                )
                continue

            if (
                rank_context == "normal"
                and query_type == "name"
                and _metadata_has_formula(md)
                and _formula_conflicts_metadata(md, props.get("MolecularFormula"))
                and not _inchikey_matches_metadata(md, str(ikey) if ikey else None)
                and syn_score < 0.26
            ):
                ranked.append(
                    {
                        "cid": int(cid),
                        "score": -1.0,
                        "reasons": ["rejected_formula_metadata_conflict_weak_synonym", *syn_rs[:1]],
                        "synonym_evidence": syn_score,
                        "SMILES": smi,
                        "INCHI": str(inchi).strip() if inchi else None,
                        "INCHIKEY": str(ikey).strip() if ikey else None,
                        "IUPACName": props.get("IUPACName"),
                        "MolecularFormula": props.get("MolecularFormula"),
                        "MolecularWeight": props.get("MolecularWeight"),
                    }
                )
                continue

            score = 0.0
            reasons: list[str] = []
            if _inchikey_matches_metadata(md, str(ikey) if ikey else None):
                score = 0.97
                reasons.append("metadata_inchikey_agreement")
            else:
                if query_type == "cas":
                    score += 0.88
                    reasons.append("cas_xref")
                elif meta_name:
                    score += 0.40
                    reasons.append("metadata_name")
                elif path_weak:
                    score += 0.20
                    reasons.append("path_name")
                else:
                    score += 0.35
                    reasons.append("name_other")

                iupac = str(props.get("IUPACName") or "")
                exact, contained = _exact_or_contained_name_match(query, iupac)
                if exact:
                    score += 0.42
                    reasons.append("exact_normalized_match")
                elif contained:
                    score += 0.26
                    reasons.append("substring_name_match")

                syn_add, syn_lab = _inorganic_synonym_bonus(query, iupac)
                if syn_add > 0:
                    score += syn_add
                    if syn_lab:
                        reasons.append(syn_lab)

                reasons.extend(syn_rs)
                score += min(0.44, syn_score)

                if _formula_matches_metadata(md, props.get("MolecularFormula")):
                    score += 0.22
                    reasons.append("formula_metadata_match")

                if (
                    rank_context == "normal"
                    and query_type == "name"
                    and _metadata_has_formula(md)
                    and _formula_conflicts_metadata(md, props.get("MolecularFormula"))
                    and not _inchikey_matches_metadata(md, str(ikey) if ikey else None)
                ):
                    score -= 0.12
                    reasons.append("formula_metadata_conflict_penalty")

                sim = _jaccard_tokens(q_tokens, _tokenize_name(iupac))
                if sim > 0:
                    cap = 0.10 if syn_score >= 0.20 else 0.18
                    score += min(cap, sim * (0.16 if syn_score >= 0.20 else 0.22))
                    reasons.append(f"name_overlap:{sim:.2f}")

                if props.get("IsomericSMILES"):
                    score += 0.03
                    reasons.append("has_isomeric_smiles")
                if ikey:
                    score += 0.02
                    reasons.append("has_inchikey")
                if _MIXTURE_HINT_RE.search(query):
                    score -= 0.20
                    reasons.append("mixture_hint_penalty")

            ranked.append(
                {
                    "cid": int(cid),
                    "score": max(0.0, min(1.0, score)),
                    "reasons": reasons,
                    "synonym_evidence": syn_score,
                    "SMILES": smi,
                    "INCHI": str(inchi).strip() if inchi else None,
                    "INCHIKEY": str(ikey).strip() if ikey else None,
                    "IUPACName": props.get("IUPACName"),
                    "MolecularFormula": props.get("MolecularFormula"),
                    "MolecularWeight": props.get("MolecularWeight"),
                }
            )

        ranked = [r for r in ranked if float(r.get("score", -99)) >= 0]
        if not ranked:
            return None, [], None
        ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top = ranked[0]
        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
        margin = max(0.0, float(top["score"]) - second_score)

        mc = 0.85
        mm = 0.15
        top_reasons = list(top.get("reasons") or [])
        top_syn = float(top.get("synonym_evidence") or 0.0)
        if rank_context == "formula_fallback":
            mc, mm = 0.93, 0.22
        elif "metadata_inchikey_agreement" in top_reasons:
            mc, mm = 0.70, 0.05
        elif query_type == "cas":
            mc, mm = 0.82, 0.08 if len(ranked) == 1 else 0.12
        elif path_weak:
            mc, mm = 0.91, 0.18
        elif meta_name and single_cid:
            mc, mm = 0.74, 0.06
            if top_syn >= 0.34:
                mc, mm = 0.72, 0.05
        elif meta_name:
            mc, mm = 0.82, 0.12
            if top_syn >= 0.34 and "formula_metadata_match" in top_reasons:
                mc, mm = 0.80, 0.10

        ok = float(top["score"]) >= mc and margin >= mm
        if rank_context == "formula_fallback":
            ok = ok and (
                "formula_metadata_match" in top_reasons
                or "metadata_inchikey_agreement" in top_reasons
                or top_syn >= 0.28
            )

        via_out = query_type
        if rank_context == "formula_fallback":
            via_out = "formula_fallback"

        decision = {
            "status": "resolved" if ok else "ambiguous",
            "confidence": round(float(top["score"]), 4),
            "margin": round(margin, 4),
            "threshold_conf": mc,
            "threshold_margin": mm,
            "selected_cid": int(top["cid"]),
            "rank_context": rank_context,
        }
        resolved = {
            "SMILES": top.get("SMILES"),
            "INCHI": top.get("INCHI"),
            "INCHIKEY": top.get("INCHIKEY"),
            "pubchem_cid": int(top["cid"]),
            "pubchem_iupac_name": top.get("IUPACName"),
            "pubchem_formula": top.get("MolecularFormula"),
            "pubchem_mw": top.get("MolecularWeight"),
            "pubchem_via": via_out,
            "pubchem_query": query,
            "pubchem_confidence": round(float(top["score"]), 4),
            "pubchem_margin": round(margin, 4),
        }
        if not ok:
            return None, ranked[:5], decision
        return resolved, ranked[:5], decision

    for c in cas_candidates:
        query = c["query"]
        bump("cas_attempt")
        cids, cerr = fetch_cids_by_cas(query, delay_s=delay_s)
        if cerr and _is_transport_pubchem_error(cerr):
            transport_fail_seen = True
        if cids:
            got_pubchem_payload = True
        attempt: dict[str, Any] = {
            "query_type": "cas",
            "query": query,
            "source": c["source"],
            "n_cids": len(cids),
            "top_cids": cids[:cid_cap],
        }
        if cerr:
            attempt["http_error"] = cerr
        attempts.append(attempt)
        if not cids:
            continue
        resolved, ranked, decision = rank_cids(query, "cas", c["source"], cids)
        if ranked:
            attempt["ranked_candidates"] = [
                {
                    "cid": int(x["cid"]),
                    "score": x["score"],
                    "reasons": x["reasons"],
                    "iupac_name": x.get("IUPACName"),
                    "molecular_formula": x.get("MolecularFormula"),
                    "has_smiles": bool(x.get("SMILES")),
                }
                for x in ranked
            ]
        if decision:
            attempt["decision"] = decision
        if resolved:
            bump("cas_resolved")
            return resolved, attempts, False

    for c in name_candidates:
        query = c["query"]
        bump("name_attempt")
        if c["source"].startswith("path:"):
            bump("filename_attempt")
        cids, cerr = fetch_cids_by_name(query, delay_s=delay_s)
        if cerr and _is_transport_pubchem_error(cerr):
            transport_fail_seen = True
        if cids:
            got_pubchem_payload = True
        attempt = {
            "query_type": "name",
            "query": query,
            "source": c["source"],
            "n_cids": len(cids),
            "top_cids": cids[:cid_cap],
        }
        if cerr:
            attempt["http_error"] = cerr
        attempts.append(attempt)
        if not cids:
            continue
        resolved, ranked, decision = rank_cids(query, "name", c["source"], cids)
        if ranked:
            attempt["ranked_candidates"] = [
                {
                    "cid": int(x["cid"]),
                    "score": x["score"],
                    "reasons": x["reasons"],
                    "iupac_name": x.get("IUPACName"),
                    "molecular_formula": x.get("MolecularFormula"),
                    "has_smiles": bool(x.get("SMILES")),
                }
                for x in ranked
            ]
        if decision:
            attempt["decision"] = decision
        if resolved:
            bump("name_resolved")
            if c["source"].startswith("path:"):
                bump("filename_resolved")
            return resolved, attempts, False

    # Optional same-formula fallback: many NIST names are non-IUPAC; formula narrows candidates.
    mf_md = _get_first_nonempty(md, FORMULA_FIELD_KEYS)
    if mf_md:
        cids_ff, ferr = fetch_cids_by_fastformula(mf_md, delay_s=delay_s)
        if ferr and _is_transport_pubchem_error(ferr):
            transport_fail_seen = True
        if cids_ff:
            got_pubchem_payload = True
        if cids_ff and 1 <= len(cids_ff) <= 15:
            bump("fastformula_attempt")
            q_ff = _get_first_nonempty(md, NAME_FIELD_KEYS) or str(mf_md)
            attempt_ff: dict[str, Any] = {
                "query_type": "name",
                "query": q_ff,
                "source": "meta:formula_fallback:fastformula",
                "n_cids": len(cids_ff),
                "top_cids": cids_ff[:cid_cap],
                "via": "fastformula",
            }
            if ferr:
                attempt_ff["http_error"] = ferr
            resolved_ff, ranked_ff, decision_ff = rank_cids(
                q_ff,
                "name",
                "meta:formula_fallback",
                cids_ff,
                rank_context="formula_fallback",
            )
            if ranked_ff:
                attempt_ff["ranked_candidates"] = [
                    {
                        "cid": int(x["cid"]),
                        "score": x["score"],
                        "reasons": x["reasons"],
                        "synonym_evidence": x.get("synonym_evidence"),
                        "iupac_name": x.get("IUPACName"),
                        "molecular_formula": x.get("MolecularFormula"),
                        "has_smiles": bool(x.get("SMILES")),
                    }
                    for x in ranked_ff
                ]
            if decision_ff:
                attempt_ff["decision"] = decision_ff
            attempts.append(attempt_ff)
            if resolved_ff:
                bump("fastformula_resolved")
                return resolved_ff, attempts, False

    net_transport = bool(transport_fail_seen and not got_pubchem_payload)
    return None, attempts, net_transport


def cache_key(md: dict[str, Any], path_hint: str | None) -> str:
    """
    Cache key for PubChem enrichment (must match what the resolver can use).

    **Does not include** the spectrum filename leaf: the same compound often appears
    in many NIST index rows with different ``source_path``s; keying on leaf would
    force a separate PubChem round-trip per file. Path-derived CAS is included via
    ``cas_and_name_from_path_hint`` so it stays consistent with
    ``resolve_pubchem_phase1`` / ``_build_query_candidates``.
    """
    cas_meta = normalize_cas(_get_first_nonempty(md, CAS_FIELD_KEYS))
    cas_path, _ = cas_and_name_from_path_hint(path_hint)
    cas_use = cas_meta or cas_path or ""
    title = _get_first_nonempty(md, NAME_FIELD_KEYS)[:120]
    return json.dumps({"cas": cas_use, "title": title}, sort_keys=True)


def preview_pubchem_queries(md: dict[str, Any], path_hint: str | None) -> dict[str, Any]:
    """Dry-run: CAS/name candidates and metadata aliases (no HTTP)."""
    return extract_pubchem_query_bundle(md, path_hint)


def _has_struct_fields(d: dict[str, Any] | None) -> bool:
    if not isinstance(d, dict):
        return False
    smi = str(d.get("SMILES") or "").strip()
    inchi = str(d.get("INCHI") or "").strip()
    return bool(smi or inchi)


def _cache_entry(
    *,
    status: str,
    resolved: dict[str, Any] | None,
    attempts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "cache_schema": CACHE_SCHEMA,
        "lookup_strategy": LOOKUP_STRATEGY,
        "status": status,
        "resolved": dict(resolved or {}),
        "pubchem_attempts": list(attempts or [])[:12],
    }


def _parse_cache_entry(
    raw: Any,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], bool]:
    """
    Parse cache record into ``(resolved, status, attempts, current_schema)``.

    Legacy cache payloads are treated as stale and can be retried.
    """
    if not isinstance(raw, dict):
        return {}, "unresolved", [], False

    if "cache_schema" in raw or "lookup_strategy" in raw or "status" in raw:
        resolved = raw.get("resolved")
        if not isinstance(resolved, dict):
            resolved = {}
        status = str(raw.get("status") or "").strip().lower() or (
            "resolved" if _has_struct_fields(resolved) else "unresolved"
        )
        attempts = raw.get("pubchem_attempts")
        if not isinstance(attempts, list):
            attempts = []
        is_current = int(raw.get("cache_schema") or -1) == CACHE_SCHEMA and str(raw.get("lookup_strategy") or "") == LOOKUP_STRATEGY
        return resolved, status, attempts, is_current

    # Legacy format: direct resolved payload or empty dict.
    status = "resolved" if _has_struct_fields(raw) else "unresolved"
    attempts = raw.get("pubchem_attempts") if isinstance(raw.get("pubchem_attempts"), list) else []
    return dict(raw), status, attempts, False


def enrich_metadata_pubchem(
    md: dict[str, Any],
    *,
    path_hint: str | None,
    cache: dict[str, Any],
    delay_s: float = 0.25,
    offline_cache_only: bool = False,
    metrics: dict[str, int] | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Copy ``md`` and attach PubChem SMILES/InChI when missing.

    Returns ``(new_md, status)`` where ``status`` is one of:
    ``cache_hit``, ``offline_miss``, ``network_resolved``, ``network_miss``, ``network_ambiguous``, ``network_error``.
    """
    out = dict(md)
    key = cache_key(md, path_hint)

    if key in cache:
        resolved_hit, status_hit, attempts_hit, current_hit = _parse_cache_entry(cache[key])
        if attempts_hit:
            out["pubchem_attempts"] = attempts_hit[:12]
        if status_hit == "resolved":
            out.update({k: v for k, v in resolved_hit.items() if v is not None and str(v).strip() != ""})
            return out, "cache_hit"
        if status_hit in {"unresolved", "miss", "ambiguous"} and current_hit:
            if status_hit == "ambiguous":
                out["pubchem_via"] = "ambiguous"
                out["pubchem_query"] = str((resolved_hit or {}).get("pubchem_query") or "")
            return out, "cache_hit"
        # Legacy or stale unresolved entries should be retried with current strategy.
        if metrics is not None:
            metrics["stale_cache_retry"] = metrics.get("stale_cache_retry", 0) + 1

    if offline_cache_only:
        cache[key] = _cache_entry(status="miss", resolved=None, attempts=[])
        return out, "offline_miss"

    resolved, attempts, transport_failed = resolve_pubchem_phase1(
        md=out, path_hint=path_hint, delay_s=delay_s, metrics=metrics
    )
    if attempts:
        out["pubchem_attempts"] = attempts[:12]

    if transport_failed and not resolved:
        cache[key] = _cache_entry(status="unresolved", resolved=None, attempts=attempts)
        return out, "network_error"

    if resolved:
        for k, v in resolved.items():
            if v is not None and str(v).strip():
                out[k] = v
        cache[key] = _cache_entry(status="resolved", resolved=resolved, attempts=attempts)
        return out, "network_resolved"

    status_out = "unresolved"
    if attempts and any((a.get("decision") or {}).get("status") == "ambiguous" for a in attempts):
        out["pubchem_via"] = "ambiguous"
        out["pubchem_query"] = str((attempts[-1] or {}).get("query") or "")
        status_out = "ambiguous"
    cache[key] = _cache_entry(
        status=status_out,
        resolved={"pubchem_query": out.get("pubchem_query")} if status_out == "ambiguous" else None,
        attempts=attempts,
    )
    if status_out == "ambiguous":
        return out, "network_ambiguous"
    return out, "network_miss"


def apply_canonical_structure_to_cache(
    cache: dict[str, Any], key: str, md_work: dict[str, Any]
) -> None:
    """
    After RDKit canonicalization, write the final ``SMILES`` (and key structure fields)
    into the cache entry for reproducible re-runs. Supports both new schema and legacy flat dicts.
    """
    smi = str(md_work.get("SMILES") or "").strip()
    if not smi:
        return
    raw = cache.get(key)
    if not isinstance(raw, dict) or not raw:
        return
    resolved, status, attempts, _ = _parse_cache_entry(raw)
    if status not in {"resolved", "ambiguous"} and not _has_struct_fields(resolved):
        return
    new_res = dict(resolved)
    new_res["SMILES"] = smi
    for k in (
        "INCHI",
        "INCHIKEY",
        "pubchem_cid",
        "pubchem_iupac_name",
        "pubchem_via",
        "pubchem_query",
        "pubchem_confidence",
        "pubchem_margin",
        "pubchem_formula",
        "pubchem_mw",
    ):
        v = md_work.get(k)
        if v is not None and str(v).strip() != "":
            new_res[k] = v
    st = "resolved" if _has_struct_fields(new_res) else status
    cache[key] = _cache_entry(status=st, resolved=new_res, attempts=attempts)


def validate_structure_smiles(md: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Optional RDKit check that ``SMILES`` parses before Mordred.

    Returns ``(True, None)`` when there is no SMILES or parse succeeds;
    ``(False, reason)`` when RDKit rejects the SMILES.
    """
    smi = str(md.get("SMILES") or "").strip()
    if not smi:
        return True, None
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False, "rdkit_parse_failed"
        return True, None
    except Exception as exc:
        return False, f"rdkit_error:{type(exc).__name__}"


def load_json_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_json_cache(path: Path, data: dict[str, Any]) -> None:
    """
    Persist the PubChem lookup cache as JSON.

    Writes to a ``*.json.tmp`` sibling then ``os.replace`` into place so a crash
    mid-write is less likely to leave a truncated/corrupt main file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main_cli() -> int:
    """CLI entrypoint for ``python -m ml.pubchem_structure_lookup``."""
    import argparse

    ap = argparse.ArgumentParser(description="PubChem PUG REST lookup (same stack as structural_fg_svm build-dataset).")
    ap.add_argument("--cas", default="", help="Optional CAS registry number")
    ap.add_argument("--title", default="", help="Optional compound title/name")
    ap.add_argument("--path", default="", help="Optional filename/path hint")
    ap.add_argument("--pubchem-delay", type=float, default=0.25)
    ap.add_argument("--preview-only", action="store_true", help="Print candidate queries only (no HTTP)")
    args = ap.parse_args()
    md: dict[str, Any] = {}
    if str(args.cas).strip():
        md["cas"] = str(args.cas).strip()
    if str(args.title).strip():
        t = str(args.title).strip()
        md["title"] = t
        md["name"] = t
    path = str(args.path).strip() or None
    bundle = extract_pubchem_query_bundle(md, path)
    if args.preview_only:
        print(json.dumps(bundle, indent=2))
        return 0
    resolved, attempts, transport_failed = resolve_pubchem_phase1(
        md=md, path_hint=path, delay_s=float(args.pubchem_delay)
    )
    print(
        json.dumps(
            {"preview": bundle, "resolved": resolved, "transport_failed": transport_failed, "attempts": attempts[:16]},
            indent=2,
        )
    )
    return 0 if resolved else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
