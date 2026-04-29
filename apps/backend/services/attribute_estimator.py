"""BanusNas — Soft Biometric Attribute Estimator.

Estimates physical attributes from person crops to improve recognition accuracy:
  - Gender (male/female) — via InsightFace genderage model when face visible
  - Age group (child/young_adult/adult/middle_aged/senior)
  - Build (slim/medium/large) — from body aspect ratio + area heuristics
  - Height category (short/medium/tall) — from bbox height relative to frame
  - Posture (standing/walking/sitting/crouching) — from bbox aspect ratio
  - Upper/lower clothing colour — from colour histogram of crop regions

Stable attributes (gender, age_group, build) are learned over time on NamedObject.
Transient attributes (clothing, posture) help same-session matching.

The attribute match score is used as a multiplier (0.70–1.15) on embedding similarity.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Colour names for clothing ──

_COLOUR_RANGES_HSV = [
    # (name, H_low, H_high, S_min, V_min)
    ("red",       0,   10, 70, 50),
    ("orange",   10,   25, 70, 50),
    ("yellow",   25,   35, 70, 50),
    ("green",    35,   85, 40, 40),
    ("blue",     85,  130, 40, 40),
    ("purple",  130,  160, 30, 40),
    ("red",     160,  180, 70, 50),   # wraparound red
]

_ACHROMATIC_NAMES = [
    # (name, V_low, V_high, S_max)
    ("black",   0,  60, 60),
    ("grey",   60, 170, 50),
    ("white", 170, 256, 60),
]


def _dominant_colour(region: np.ndarray) -> str:
    """Return the dominant colour name of a BGR image region."""
    if region.size == 0 or region.shape[0] < 4 or region.shape[1] < 4:
        return "unknown"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Check achromatic first (low saturation)
    achro_mask = s < 50
    if np.count_nonzero(achro_mask) > 0.55 * achro_mask.size:
        mean_v = float(np.mean(v[achro_mask]))
        for name, v_low, v_high, _ in _ACHROMATIC_NAMES:
            if v_low <= mean_v < v_high:
                return name
        return "grey"

    # Chromatic: bin hue values where S and V are sufficient
    chroma_mask = (s >= 40) & (v >= 40)
    if np.count_nonzero(chroma_mask) < 0.15 * chroma_mask.size:
        # Too few chromatic pixels — report as achromatic
        mean_v = float(np.mean(v))
        if mean_v < 60:
            return "black"
        if mean_v > 170:
            return "white"
        return "grey"

    hue_vals = h[chroma_mask]
    best_name = "unknown"
    best_count = 0
    for name, h_low, h_high, s_min, v_min in _COLOUR_RANGES_HSV:
        count = int(np.count_nonzero((hue_vals >= h_low) & (hue_vals < h_high)))
        if count > best_count:
            best_count = count
            best_name = name

    return best_name


# ── Attribute estimation ──

class PersonAttributes:
    """Estimated attributes for a single person detection."""

    __slots__ = (
        "gender", "gender_conf",
        "age", "age_group",
        "build", "height_ratio",
        "posture", "upper_color", "lower_color",
        "hair_color", "skin_tone",
    )

    def __init__(self):
        self.gender: Optional[str] = None          # "male" | "female"
        self.gender_conf: float = 0.0
        self.age: Optional[int] = None              # estimated integer age
        self.age_group: Optional[str] = None        # "child" | "young_adult" | "adult" | "middle_aged" | "senior"
        self.build: Optional[str] = None            # "slim" | "medium" | "large"
        self.height_ratio: float = 0.0              # bbox_height / frame_height
        self.posture: Optional[str] = None          # "standing" | "walking" | "sitting" | "crouching"
        self.upper_color: Optional[str] = None
        self.lower_color: Optional[str] = None
        self.hair_color: Optional[str] = None       # "black" | "brown" | "blonde" | "red" | "grey" | "white"
        self.skin_tone: Optional[str] = None        # "light" | "medium" | "dark"

    def to_dict(self) -> dict:
        d = {}
        if self.gender:
            d["gender"] = self.gender
            d["gender_conf"] = round(self.gender_conf, 2)
        if self.age is not None:
            d["age"] = self.age
        if self.age_group:
            d["age_group"] = self.age_group
        if self.build:
            d["build"] = self.build
        if self.height_ratio > 0:
            d["height_ratio"] = round(self.height_ratio, 3)
        if self.posture:
            d["posture"] = self.posture
        if self.upper_color:
            d["upper_color"] = self.upper_color
        if self.lower_color:
            d["lower_color"] = self.lower_color
        if self.hair_color:
            d["hair_color"] = self.hair_color
        if self.skin_tone:
            d["skin_tone"] = self.skin_tone
        return d


def _age_to_group(age: int) -> str:
    if age < 16:
        return "child"
    if age < 28:
        return "young_adult"
    if age < 45:
        return "adult"
    if age < 65:
        return "middle_aged"
    return "senior"


# ── Hair colour ranges (HSV) ──
_HAIR_COLOUR_MAP = [
    # (name, H_low, H_high, S_min, S_max, V_min, V_max)
    ("red",     0,   15, 80, 255, 60, 200),
    ("blonde", 15,   30, 40, 180, 140, 255),
    ("brown",   8,   25, 30, 180, 40, 140),
    ("red",   160,  180, 60, 255, 60, 200),
]


def _detect_hair_color(crop: np.ndarray, face_bbox: tuple | None = None) -> Optional[str]:
    """Detect hair colour from the top region of a person crop.

    Uses the head area above the face if face_bbox is available,
    otherwise the top 12% of the crop.
    """
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 15:
        return None

    if face_bbox is not None:
        # Hair is above the face
        fy1 = max(0, int(face_bbox[1]))
        hair_top = max(0, fy1 - int((face_bbox[3] - face_bbox[1]) * 0.3))
        hair_region = crop[hair_top:fy1, :]
    else:
        # Top 12% of the person crop (head area)
        hair_region = crop[0:max(4, int(ch * 0.12)), :]

    if hair_region.size == 0 or hair_region.shape[0] < 3 or hair_region.shape[1] < 3:
        return None

    hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Check for achromatic hair (black, grey, white) — low saturation
    achro_mask = s < 45
    achro_ratio = np.count_nonzero(achro_mask) / max(achro_mask.size, 1)

    if achro_ratio > 0.55:
        mean_v = float(np.mean(v[achro_mask])) if np.any(achro_mask) else float(np.mean(v))
        if mean_v < 55:
            return "black"
        if mean_v > 185:
            return "white"
        return "grey"

    # Chromatic hair — find best matching colour
    chroma_mask = (s >= 30) & (v >= 30)
    if np.count_nonzero(chroma_mask) < 0.12 * chroma_mask.size:
        mean_v = float(np.mean(v))
        return "black" if mean_v < 70 else "brown"

    hue_vals = h[chroma_mask]
    sat_vals = s[chroma_mask]
    val_vals = v[chroma_mask]

    best_name = "brown"  # default for ambiguous
    best_count = 0
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in _HAIR_COLOUR_MAP:
        count = int(np.count_nonzero(
            (hue_vals >= h_lo) & (hue_vals < h_hi) &
            (sat_vals >= s_lo) & (sat_vals <= s_hi) &
            (val_vals >= v_lo) & (val_vals <= v_hi)
        ))
        if count > best_count:
            best_count = count
            best_name = name

    return best_name


# ── Skin tone ranges (HSV-based) ──
_SKIN_HUE_LOW, _SKIN_HUE_HIGH = 5, 25
_SKIN_SAT_MIN = 30
_SKIN_VAL_MIN = 50


def _detect_skin_tone(region: np.ndarray) -> Optional[str]:
    """Detect skin tone from a face or exposed-skin region.

    Returns 'light', 'medium', or 'dark'.
    """
    if region.size == 0 or region.shape[0] < 8 or region.shape[1] < 8:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Skin-coloured pixel mask
    skin_mask = (
        (h >= _SKIN_HUE_LOW) & (h <= _SKIN_HUE_HIGH) &
        (s >= _SKIN_SAT_MIN) & (v >= _SKIN_VAL_MIN)
    )
    skin_count = np.count_nonzero(skin_mask)
    if skin_count < 0.08 * skin_mask.size:
        return None  # Not enough skin pixels

    mean_v = float(np.mean(v[skin_mask]))
    mean_s = float(np.mean(s[skin_mask]))

    # Classify by brightness + saturation
    if mean_v > 170 and mean_s < 80:
        return "light"
    if mean_v < 100 or mean_s > 120:
        return "dark"
    return "medium"


def estimate_person_attributes(
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    face_data=None,
) -> PersonAttributes:
    """Estimate soft biometric attributes from a person crop and bounding box.

    Parameters
    ----------
    crop : BGR person crop image
    bbox : (x1, y1, x2, y2) in full-frame coordinates
    frame_shape : (frame_height, frame_width)
    face_data : InsightFace Face object (optional) — for gender/age when available

    Returns
    -------
    PersonAttributes with estimated fields
    """
    attrs = PersonAttributes()
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    frame_h, frame_w = frame_shape[:2]

    if box_w < 10 or box_h < 10:
        return attrs

    # ── Gender & age from InsightFace face data ──
    if face_data is not None:
        try:
            if hasattr(face_data, "gender") and face_data.gender is not None:
                # InsightFace: gender=0 female, gender=1 male
                attrs.gender = "male" if int(face_data.gender) == 1 else "female"
                attrs.gender_conf = 0.85  # InsightFace doesn't expose confidence
            if hasattr(face_data, "age") and face_data.age is not None:
                attrs.age = int(face_data.age)
                attrs.age_group = _age_to_group(attrs.age)
        except (AttributeError, TypeError, ValueError):
            pass

    # ── Head-to-body ratio child heuristic ──
    # Children have proportionally larger heads than adults:
    #   - Toddlers (~2yr): head ≈ 1/4 of total height (ratio ≈ 0.25)
    #   - Children (~6yr): head ≈ 1/6 of total height (ratio ≈ 0.17)
    #   - Adults:          head ≈ 1/7.5 of total height (ratio ≈ 0.13)
    # When a face is detected within a full-body crop, this ratio is a
    # reliable cue even when InsightFace's age regressor is biased (it was
    # trained mostly on adult faces and tends to over-estimate children's age).
    head_body_ratio: Optional[float] = None
    if face_data is not None and hasattr(face_data, "bbox"):
        try:
            fb = face_data.bbox  # crop-local coords
            face_h_px = float(fb[3]) - float(fb[1])
            if face_h_px > 5 and box_h > 40:
                # Approximate head height ≈ 1.35 × face (face bbox excludes hair/chin)
                head_h_est = face_h_px * 1.35
                head_body_ratio = head_h_est / float(box_h)
        except (TypeError, IndexError, ValueError):
            head_body_ratio = None

    if head_body_ratio is not None:
        # Require posture to be a full standing/walking body — otherwise the
        # ratio is meaningless (sitting / occluded subjects).
        is_full_body = (box_h / max(box_w, 1)) > 1.8
        if is_full_body:
            if head_body_ratio >= 0.20:
                # Strong child signal — override face-age if it disagreed.
                attrs.age_group = "child"
                if attrs.age is None:
                    attrs.age = 8
            elif head_body_ratio >= 0.165 and attrs.age_group in (None, "young_adult"):
                # Borderline child / pre-teen — only override if face age was
                # ambiguous (None) or low-end "young_adult".
                attrs.age_group = "child"
                if attrs.age is None:
                    attrs.age = 12

    # ── Fallback child heuristic when no face is available ──
    # If no face was detected (common for small/distant subjects or kids
    # facing away) but the bbox is clearly a full-body standing person AND
    # is unusually short relative to the frame, flag as a likely child.
    # This is intentionally conservative — only fires when the bbox is in
    # the lower 60% of the frame (i.e., a near subject, not a distant one
    # that just happens to be small in pixels).
    if attrs.age_group is None and face_data is None:
        is_full_body = (box_h / max(box_w, 1)) > 1.9
        height_ratio = box_h / max(frame_h, 1)
        bbox_bottom_ratio = y2 / max(frame_h, 1)
        if (
            is_full_body
            and height_ratio < 0.35
            and bbox_bottom_ratio > 0.55
        ):
            attrs.age_group = "child"

    # ── Build from aspect ratio and area ──
    aspect_ratio = box_w / max(box_h, 1)
    relative_area = (box_w * box_h) / max(frame_h * frame_w, 1)

    # Wider aspect ratio (relative to height) suggests larger build
    if aspect_ratio > 0.55:
        attrs.build = "large"
    elif aspect_ratio > 0.38:
        attrs.build = "medium"
    else:
        attrs.build = "slim"

    # ── Height from bbox relative to frame ──
    attrs.height_ratio = box_h / max(frame_h, 1)

    # ── Posture from aspect ratio ──
    inv_ar = box_h / max(box_w, 1)  # height/width
    if inv_ar > 2.8:
        attrs.posture = "standing"
    elif inv_ar > 1.8:
        attrs.posture = "walking"
    elif inv_ar > 1.2:
        attrs.posture = "sitting"
    else:
        attrs.posture = "crouching"

    # ── Clothing colours ──
    ch, cw = crop.shape[:2]
    if ch >= 20 and cw >= 10:
        # Upper body: top 45% (skip top 5% which is often head/hair)
        upper_start = max(0, int(ch * 0.08))
        upper_end = int(ch * 0.45)
        upper_region = crop[upper_start:upper_end, :]
        attrs.upper_color = _dominant_colour(upper_region)

        # Lower body: bottom 45% (skip very bottom 5% which may be ground)
        lower_start = int(ch * 0.50)
        lower_end = max(lower_start + 1, int(ch * 0.95))
        lower_region = crop[lower_start:lower_end, :]
        attrs.lower_color = _dominant_colour(lower_region)

    # ── Hair colour ──
    face_bbox_local = None
    if face_data is not None and hasattr(face_data, "bbox"):
        try:
            fb = face_data.bbox
            # face_data.bbox is in crop-local coords
            face_bbox_local = (int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]))
        except (TypeError, IndexError):
            pass
    attrs.hair_color = _detect_hair_color(crop, face_bbox_local)

    # ── Skin tone ──
    if face_data is not None and face_bbox_local is not None:
        # Use face region for skin tone detection
        fy1 = max(0, face_bbox_local[1])
        fy2 = min(ch, face_bbox_local[3])
        fx1 = max(0, face_bbox_local[0])
        fx2 = min(cw, face_bbox_local[2])
        if fy2 > fy1 + 5 and fx2 > fx1 + 5:
            face_region = crop[fy1:fy2, fx1:fx2]
            attrs.skin_tone = _detect_skin_tone(face_region)
    if attrs.skin_tone is None and ch >= 30:
        # Fallback: sample the upper-center area (neck/face vicinity)
        center_x = cw // 2
        region_w = max(8, cw // 4)
        region = crop[0:int(ch * 0.15), max(0, center_x - region_w):min(cw, center_x + region_w)]
        attrs.skin_tone = _detect_skin_tone(region)

    return attrs


# ── Stable attribute learning ──

AGE_GROUP_ORDER = ["child", "young_adult", "adult", "middle_aged", "senior"]


def merge_stable_attributes(
    existing: Optional[dict],
    new_attrs: PersonAttributes,
) -> dict:
    """Merge newly estimated attributes into a stored attribute profile.

    Uses running counts per attribute value to determine consensus.
    Only updates when the new detection has a non-None value.
    """
    profile = dict(existing) if existing else {}

    def _update_categorical(key: str, value: Optional[str], conf: float = 1.0):
        if not value:
            return
        # Never overwrite manually-set attributes with auto-detected values
        if profile.get(f"_{key}_manual"):
            return
        votes = profile.get(f"_{key}_votes", {})
        votes[value] = votes.get(value, 0.0) + conf
        profile[f"_{key}_votes"] = votes
        # Winner = highest vote
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        profile[key] = winner
        profile[f"{key}_confidence"] = round(votes[winner] / max(total, 1e-6), 2)
        profile[f"{key}_samples"] = int(sum(votes.values()))

    # Gender (weighted by per-detection confidence)
    _update_categorical("gender", new_attrs.gender, new_attrs.gender_conf)

    # Age group
    _update_categorical("age_group", new_attrs.age_group)

    # Build
    _update_categorical("build", new_attrs.build)

    # Hair colour
    _update_categorical("hair_color", new_attrs.hair_color)

    # Skin tone
    _update_categorical("skin_tone", new_attrs.skin_tone)

    # Height ratio — running average
    if new_attrs.height_ratio > 0:
        n = profile.get("_height_n", 0)
        avg = profile.get("height_ratio", 0.0)
        new_avg = (avg * n + new_attrs.height_ratio) / (n + 1)
        profile["height_ratio"] = round(new_avg, 4)
        profile["_height_n"] = n + 1

    return profile


def get_display_attributes(profile: Optional[dict]) -> dict:
    """Return structured display-safe attributes.

    Returns dict of ``{key: {value, confidence, samples}}`` for categorical
    attributes, filtering out internal vote tallies.
    """
    if not profile:
        return {}
    result = {}
    for key in ("gender", "age_group", "build", "hair_color", "skin_tone"):
        value = profile.get(key)
        if value:
            result[key] = {
                "value": value,
                "confidence": profile.get(f"{key}_confidence", 0),
                "samples": profile.get(f"{key}_samples", 0),
                "manual": profile.get(f"_{key}_manual", False),
            }
    hr = profile.get("height_ratio")
    if hr and hr > 0:
        result["height_estimate"] = {
            "value": "tall" if hr > 0.55 else "short" if hr < 0.30 else "average",
            "confidence": min(profile.get("_height_n", 1) / 10.0, 1.0),
            "samples": profile.get("_height_n", 0),
        }
    return result


# ── Attribute matching scorer ──

def compute_attribute_multiplier(
    detected_attrs: PersonAttributes,
    stored_profile: Optional[dict],
) -> float:
    """Compute a confidence multiplier (0.70–1.15) based on attribute agreement.

    Returns 1.0 if no stored profile or nothing to compare.
    """
    if not stored_profile:
        return 1.0

    multiplier = 1.0

    # Gender match/mismatch (strongest signal)
    stored_gender = stored_profile.get("gender")
    gender_conf = stored_profile.get("gender_confidence", 0)
    gender_manual = stored_profile.get("_gender_manual", False)
    if stored_gender and detected_attrs.gender and gender_conf >= 0.6:
        if detected_attrs.gender == stored_gender:
            multiplier *= 1.06
        else:
            # Manual gender: strong penalty. Auto-estimated: mild penalty
            # (auto-estimated gender from small CCTV faces is often wrong)
            multiplier *= 0.65 if gender_manual else 0.85

    # Age group match/mismatch
    stored_age = stored_profile.get("age_group")
    age_conf = stored_profile.get("age_group_confidence", 0)
    age_manual = stored_profile.get("_age_group_manual", False)
    if stored_age and detected_attrs.age_group and age_conf >= 0.5:
        stored_idx = AGE_GROUP_ORDER.index(stored_age) if stored_age in AGE_GROUP_ORDER else -1
        det_idx = AGE_GROUP_ORDER.index(detected_attrs.age_group) if detected_attrs.age_group in AGE_GROUP_ORDER else -1
        if stored_idx >= 0 and det_idx >= 0:
            gap = abs(stored_idx - det_idx)
            if gap == 0:
                multiplier *= 1.04
            elif gap == 1:
                pass  # Adjacent groups — neutral
            elif gap >= 3:
                multiplier *= 0.50 if age_manual else 0.60  # Extreme mismatch
            elif gap >= 2:
                multiplier *= 0.60 if age_manual else 0.72  # Large age mismatch

    # Build match/mismatch
    stored_build = stored_profile.get("build")
    build_conf = stored_profile.get("build_confidence", 0)
    if stored_build and detected_attrs.build and build_conf >= 0.5:
        build_order = ["slim", "medium", "large"]
        stored_bi = build_order.index(stored_build) if stored_build in build_order else -1
        det_bi = build_order.index(detected_attrs.build) if detected_attrs.build in build_order else -1
        if stored_bi >= 0 and det_bi >= 0:
            gap = abs(stored_bi - det_bi)
            if gap == 0:
                multiplier *= 1.03
            elif gap >= 2:
                multiplier *= 0.88

    # Hair colour match/mismatch — strong discriminator once learned
    stored_hair = stored_profile.get("hair_color")
    hair_conf = stored_profile.get("hair_color_confidence", 0)
    hair_samples = stored_profile.get("hair_color_samples", 0)
    if stored_hair and detected_attrs.hair_color and hair_conf >= 0.55 and hair_samples >= 3:
        if detected_attrs.hair_color == stored_hair:
            multiplier *= 1.06
        else:
            # Hair colour mismatch is a strong signal — black vs blonde is definitive
            _hair_order = ["black", "brown", "red", "blonde", "grey", "white"]
            s_hi = _hair_order.index(stored_hair) if stored_hair in _hair_order else -1
            d_hi = _hair_order.index(detected_attrs.hair_color) if detected_attrs.hair_color in _hair_order else -1
            if s_hi >= 0 and d_hi >= 0:
                gap = abs(s_hi - d_hi)
                if gap >= 3:
                    multiplier *= 0.72  # Very different (e.g. black vs blonde)
                elif gap >= 2:
                    multiplier *= 0.85  # Somewhat different
                # gap == 1 is neutral (adjacent colours are easy to confuse on CCTV)

    # Skin tone match/mismatch — stable biometric
    stored_skin = stored_profile.get("skin_tone")
    skin_conf = stored_profile.get("skin_tone_confidence", 0)
    skin_samples = stored_profile.get("skin_tone_samples", 0)
    if stored_skin and detected_attrs.skin_tone and skin_conf >= 0.55 and skin_samples >= 3:
        _tone_order = ["light", "medium", "dark"]
        s_ti = _tone_order.index(stored_skin) if stored_skin in _tone_order else -1
        d_ti = _tone_order.index(detected_attrs.skin_tone) if detected_attrs.skin_tone in _tone_order else -1
        if s_ti >= 0 and d_ti >= 0:
            gap = abs(s_ti - d_ti)
            if gap == 0:
                multiplier *= 1.05
            elif gap >= 2:
                multiplier *= 0.75  # Light vs dark is a definitive mismatch

    # Height ratio comparison (only if both have sufficient samples)
    stored_hr = stored_profile.get("height_ratio", 0)
    height_n = stored_profile.get("_height_n", 0)
    if stored_hr > 0 and detected_attrs.height_ratio > 0 and height_n >= 3:
        ratio_diff = abs(detected_attrs.height_ratio - stored_hr) / max(stored_hr, 0.01)
        if ratio_diff < 0.15:
            multiplier *= 1.02  # Very similar height
        elif ratio_diff > 0.40:
            multiplier *= 0.92  # Very different height

    # Wider clamp range — stronger floor allows attribute penalties
    # (age + gender + hair + skin) to meaningfully reject bad matches
    return max(0.50, min(1.20, multiplier))
