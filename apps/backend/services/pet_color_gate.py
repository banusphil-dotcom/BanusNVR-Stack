"""Pet appearance gates — colour-prior + margin enforcement.

The CNN cosine similarity alone has proven insufficient to distinguish cats with
very different fur colours (white Persian "Frostie" vs tortoiseshell Persian
"Tangie") because:

1.  Reference embeddings are class-imbalanced (216 vs 44) — the dominant cat's
    averaged centroid covers most of the feature space.
2.  Both cats are the same breed (Persian) so head shape / pose features dominate
    over fur colour in MobileNetV2's last pooling layer.
3.  The accept threshold (0.45) is well below the score either cat earns.

This module computes two independent appearance signals from the crop and gates
the embedding match against the candidate's stored ``attributes['color']``:

*   **white_ratio** — fraction of pixels that are bright + low-saturation (white
    fur).  >0.35 = clearly white.
*   **dominant_hue_label** — coarse colour bucket from the most common HSV hue
    bin (excluding white/black pixels).

Designed to be cheap (single 256×256 HSV pass, no model load) so it can run
inside the recognition critical path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV thresholds (OpenCV: H 0..179, S 0..255, V 0..255)
_WHITE_S_MAX = 60      # low saturation ⇒ near-grey/white
_WHITE_V_MIN = 170     # bright ⇒ white not grey
_BLACK_V_MAX = 60      # dark ⇒ black fur (ignored when bucketing hue)

# Coarse colour buckets used for canonical "color" attribute matching.
_HUE_BUCKETS = [
    ("orange", 5, 22),    # ginger / tabby orange
    ("yellow", 22, 35),   # blonde / cream
    ("green",  35, 80),
    ("blue",   80, 130),
    ("purple", 130, 160),
    ("red",    160, 179),
]

# Canonical colour family for each profile-attribute spelling we've seen in the
# field.  Mapping is many-to-one onto our gate's vocabulary:
#   white | black | tortoiseshell | tabby | orange | grey | mixed
_COLOR_NORMALISATION = {
    "white": "white", "cream": "white", "snow": "white",
    "black": "black",
    "tortoiseshell": "tortoiseshell", "tortie": "tortoiseshell",
    "calico": "tortoiseshell", "tricolor": "tortoiseshell", "tricolour": "tortoiseshell",
    "tabby": "tabby", "brown tabby": "tabby", "brown": "tabby",
    "orange": "orange", "ginger": "orange", "red": "orange",
    "grey": "grey", "gray": "grey", "silver": "grey", "blue": "grey",
    "tuxedo": "tuxedo",
    "mixed": "mixed", "multi": "mixed", "multi-colour": "mixed", "multi-color": "mixed",
}


@dataclass(frozen=True)
class PetColourSignal:
    """Cheap colour fingerprint of a pet crop."""
    white_ratio: float        # 0..1
    black_ratio: float        # 0..1
    dominant_hue: Optional[str]   # one of _HUE_BUCKETS labels or None
    dominant_hue_strength: float  # 0..1, fraction of non-white/black pixels in dominant bucket
    # Higher-level summary — same vocabulary as _COLOR_NORMALISATION values.
    family: str               # white | black | tortoiseshell | tabby | orange | grey | tuxedo | mixed
    # When True, the image is too low-contrast / desaturated (e.g. night IR
    # camera footage) for the colour family to be reliable.  ``family`` is still
    # filled in but the compatibility check should NOT veto on this signal.
    low_confidence: bool = False


def normalise_color_attr(value: Optional[str]) -> Optional[str]:
    """Map a free-form profile colour string onto the gate's vocabulary."""
    if not value:
        return None
    key = value.strip().lower()
    if key in _COLOR_NORMALISATION:
        return _COLOR_NORMALISATION[key]
    # Try first word match (e.g. "white-and-grey" → "white")
    first = key.split()[0].split("-")[0]
    return _COLOR_NORMALISATION.get(first)


def compute_colour_signal(crop: np.ndarray) -> PetColourSignal:
    """Cheap O(N) HSV summary of a pet crop.

    Strips a 10% border around the crop to avoid background contamination
    (Frigate bbox padding is generous), then computes white/black ratios and the
    dominant hue bucket among coloured pixels.
    """
    if crop is None or crop.size == 0:
        return PetColourSignal(0.0, 0.0, None, 0.0, "mixed")

    # Centre-crop 80% to suppress background
    h, w = crop.shape[:2]
    pad_h = max(1, int(h * 0.10))
    pad_w = max(1, int(w * 0.10))
    inner = crop[pad_h:h - pad_h, pad_w:w - pad_w]
    if inner.size == 0:
        inner = crop

    small = cv2.resize(inner, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = float(H.size)

    # Detect night-IR / monochrome frames: very low average saturation across
    # the whole crop ⇒ the colour family classifier is unreliable and should
    # not veto. Typical IR footage has S ≈ 5..15 everywhere.
    mean_s = float(S.mean())
    low_confidence = mean_s < 25.0

    white_mask = (S < _WHITE_S_MAX) & (V > _WHITE_V_MIN)
    black_mask = (V < _BLACK_V_MAX)
    white_ratio = float(white_mask.sum()) / total
    black_ratio = float(black_mask.sum()) / total

    coloured_mask = ~(white_mask | black_mask)
    coloured_pixels = int(coloured_mask.sum())
    dominant_hue: Optional[str] = None
    dominant_strength = 0.0
    bucket_counts: dict[str, int] = {}
    if coloured_pixels >= 50:
        H_col = H[coloured_mask]
        for label, lo, hi in _HUE_BUCKETS:
            count = int(((H_col >= lo) & (H_col < hi)).sum())
            if count:
                bucket_counts[label] = count
        if bucket_counts:
            dominant_hue, dom_count = max(bucket_counts.items(), key=lambda kv: kv[1])
            dominant_strength = dom_count / coloured_pixels

    family = _classify_family(
        white_ratio, black_ratio,
        bucket_counts, coloured_pixels, total,
    )
    return PetColourSignal(
        white_ratio=white_ratio,
        black_ratio=black_ratio,
        dominant_hue=dominant_hue,
        dominant_hue_strength=dominant_strength,
        family=family,
        low_confidence=low_confidence,
    )


def _classify_family(white_r: float, black_r: float,
                     buckets: dict[str, int],
                     coloured_pixels: int, total: float) -> str:
    coloured_r = coloured_pixels / total if total else 0.0
    orange_r = (buckets.get("orange", 0) + buckets.get("yellow", 0)) / total if total else 0.0
    # White-dominant fur (Frostie territory).
    if white_r > 0.40 and coloured_r < 0.25 and black_r < 0.10:
        return "white"
    if black_r > 0.45 and coloured_r < 0.20 and white_r < 0.10:
        return "black"
    # Tuxedo (white belly + black back, no significant colour)
    if white_r > 0.20 and black_r > 0.20 and coloured_r < 0.20:
        return "tuxedo"
    # Tortoiseshell / calico — patches of orange + dark + (often) white
    if orange_r > 0.10 and (black_r > 0.10 or white_r > 0.10):
        return "tortoiseshell"
    if orange_r > 0.30:
        return "orange"
    if buckets:
        # Largely brown / tabby — orange present but not dominant
        if orange_r > 0.05:
            return "tabby"
    if 0.10 < white_r < 0.40 and 0.10 < black_r < 0.40:
        return "grey"
    return "mixed"


# ── Compatibility matrix between profile colour and observed crop colour ──
# Returns a multiplier for the cosine score.  1.00 = no change, <1 penalises,
# 0.0 = veto.  Conservative: only veto when the contradiction is unambiguous.
_COMPATIBILITY: dict[tuple[str, str], float] = {
    # Profile == Observed → strong boost
    ("white", "white"): 1.15,
    ("black", "black"): 1.15,
    ("tortoiseshell", "tortoiseshell"): 1.10,
    ("orange", "orange"): 1.12,
    ("tabby", "tabby"): 1.05,
    ("grey", "grey"): 1.05,
    ("tuxedo", "tuxedo"): 1.08,
    # White cat seen as tuxedo (some shadow on belly) — still acceptable.
    ("white", "tuxedo"): 0.95,
    ("tuxedo", "white"): 0.85,
    # Hard incompatibilities — these are vetoes.
    ("white", "black"): 0.0,
    ("white", "tortoiseshell"): 0.0,
    ("white", "orange"): 0.10,
    ("white", "tabby"): 0.20,
    ("black", "white"): 0.0,
    ("black", "tortoiseshell"): 0.20,
    ("black", "orange"): 0.20,
    ("tortoiseshell", "white"): 0.0,
    ("tortoiseshell", "black"): 0.30,
    ("orange", "white"): 0.10,
    ("orange", "black"): 0.20,
    ("tabby", "white"): 0.20,
}


def colour_compatibility(profile_color: Optional[str],
                         signal: PetColourSignal) -> float:
    """Return a multiplier (>0 ok, ==0 veto) for ``profile_color`` against the
    observed colour signal.  Unknown profiles return 1.0 (no opinion).

    Low-confidence signals (night IR / monochrome frames) are never vetoed —
    they cap at 1.0 so the cosine match decides on its own.
    """
    fam = normalise_color_attr(profile_color)
    if not fam:
        return 1.0
    if fam == "mixed":
        return 1.0

    # ── Positive-evidence guards for high-contrast solid colours ──
    # The family classifier is conservative: a partly-shadowed white cat may
    # come back as "mixed" / "grey" / "tabby" rather than "white". Without
    # a hard requirement here, the previous logic happily tagged a clearly
    # tortoiseshell cat as "Frostie" (white) at score 0.80 because the
    # default fallback returned 0.85. Demand visible evidence of the
    # profile colour for the strongly-distinctive families. These checks
    # apply even when low_confidence is set (night IR): if you can't even
    # see white pixels in the crop, do NOT inherit a "white cat" identity.
    # The thresholds are deliberately gentle so a partly-shadowed white cat
    # still passes (white_ratio>=0.15).
    if fam == "white" and signal.white_ratio < 0.15:
        return 0.0
    if fam == "black" and signal.black_ratio < 0.15:
        return 0.0

    if signal.low_confidence:
        return 1.0  # never veto on unreliable colour read beyond the above

    if fam == "white" and signal.white_ratio < 0.30:
        # Fully-confident reading but <30% bright low-saturation pixels.
        return 0.0
    if fam == "black" and signal.black_ratio < 0.30:
        return 0.0
    if fam == "tortoiseshell":
        bright_orange = (signal.dominant_hue in ("orange", "yellow")
                         and signal.dominant_hue_strength > 0.10)
        if not bright_orange and signal.family not in ("tortoiseshell", "tabby", "orange"):
            return 0.0

    key = (fam, signal.family)
    if key in _COMPATIBILITY:
        return _COMPATIBILITY[key]
    # Default: same family unspecified above ⇒ 1.0; otherwise slight penalty.
    if fam == signal.family:
        return 1.0
    return 0.85


def is_white_cat(signal: PetColourSignal) -> bool:
    """Convenience: clearly white cat (Frostie-style)."""
    return signal.white_ratio > 0.35 and signal.family in ("white", "tuxedo")
