"""BanusNas — AI Recognition Agent.

An intelligent coordinator that orchestrates the recognition pipeline with
adaptive thresholds, context-aware routing, detection quality validation,
match coherence checking, and a remote ML-based attribute classifier.

Architecture
~~~~~~~~~~~~
The agent maintains per-camera scene context (lighting, activity level,
recognition history) and uses it to:

1. **Validate detections** — reject low-quality crops (blurry, too small,
   implausible aspect ratio) before they enter the expensive recognition
   pipeline.  Prevents inanimate objects from being "recognised".
2. **Adapt thresholds** — lower face thresholds at night / poor lighting,
   tighten them during high-activity periods to reduce false positives.
3. **Route intelligently** — if a camera rarely produces usable faces
   (e.g. high angle), prefer body ReID earlier instead of wasting time
   on face detection.
4. **Fuse attribute signals** — call a remote multi-task attribute model
   (gender/age/build) on the GPU server and combine its output with the
   existing heuristic estimator (now including hair colour and skin tone)
   for a more robust multiplier.
5. **Validate matches (persons)** — after recognition, verify the result
   makes sense by cross-checking:
   - *Cross-category sanity* — a "cat" detection can **never** be matched
     to a person profile (hard veto, no score override).
   - *Gender veto* — if the stored profile has a manually-confirmed gender
     and the detected gender disagrees with ≥70% confidence, reject.
   - *Clothing continuity* — same person on the same camera shouldn't
     change clothes within seconds.
   - *Hair colour / skin tone* — stable biometrics that shouldn't flip.
6. **Validate matches (pets/vehicles)** — CNN cosine matches are verified
   with visual plausibility checks:
   - *Category-class consistency* — YOLO "cat" matched to a vehicle named
     object is rejected unconditionally.
   - *Aspect-ratio plausibility* — a tall narrow crop can't be a pet.
   - *Crop quality gate* — tiny/blurry crops get penalised or rejected.
7. **Track recognition quality** — record hit/miss rates per camera and
   object to allow continuous tuning without manual intervention.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from services.attribute_estimator import (
    PersonAttributes,
    compute_attribute_multiplier,
    estimate_person_attributes,
)

logger = logging.getLogger(__name__)

# ── Scene context defaults ──

_LIGHTING_HISTORY_SIZE = 60  # Keep last N luminance readings per camera

# ── Detection quality thresholds ──
MIN_PERSON_CROP_AREA = 1800       # Minimum pixels (e.g. 30×60) to even attempt recognition
MIN_PERSON_ASPECT_RATIO = 0.8     # h/w — persons are taller than wide
MAX_PERSON_ASPECT_RATIO = 5.0     # h/w — reject absurdly thin vertical strips
MIN_PERSON_BLUR_SCORE = 15.0      # Laplacian variance below this = too blurry
MIN_PET_CROP_AREA = 800           # Pets can be smaller
MIN_CONFIDENCE_FOR_RECOGNITION = 0.45  # Don't attempt recognition on very low-conf detections

# ── Clothing continuity ──
CLOTHING_MEMORY_TTL = 300.0       # 5 min — how long to remember what someone was wearing


@dataclass
class CameraContext:
    """Per-camera running statistics that inform adaptive decisions."""

    camera_id: int
    # Lighting: mean luminance of recent frames (0–255)
    luminance_history: list[float] = field(default_factory=list)
    # Recognition hit rates over the last hour
    face_attempts: int = 0
    face_hits: int = 0
    body_attempts: int = 0
    body_hits: int = 0
    # Sliding window reset timestamp
    stats_window_start: float = field(default_factory=time.monotonic)
    # Track average face detection rate for this camera
    face_detect_rate: float = 0.5  # Start neutral

    @property
    def avg_luminance(self) -> float:
        if not self.luminance_history:
            return 128.0
        return sum(self.luminance_history) / len(self.luminance_history)

    @property
    def face_hit_rate(self) -> float:
        if self.face_attempts == 0:
            return 0.5  # Neutral before data
        return self.face_hits / self.face_attempts

    @property
    def body_hit_rate(self) -> float:
        if self.body_attempts == 0:
            return 0.5
        return self.body_hits / self.body_attempts

    def maybe_reset_window(self):
        """Reset stats every hour to adapt to changing conditions."""
        now = time.monotonic()
        if now - self.stats_window_start > 3600:
            # Decay rather than hard reset — keep some memory
            self.face_attempts = max(1, self.face_attempts // 4)
            self.face_hits = max(0, self.face_hits // 4)
            self.body_attempts = max(1, self.body_attempts // 4)
            self.body_hits = max(0, self.body_hits // 4)
            self.stats_window_start = now


@dataclass
class RecognitionPlan:
    """The agent's decision for how to process a person detection."""

    # Adjusted thresholds for this specific detection
    face_threshold: float = 0.28
    body_threshold: float = 0.40
    # Whether to attempt face detection at all (skip for cameras with very low face rate)
    try_face: bool = True
    # Whether to prefer body-first routing (inverted priority)
    prefer_body_first: bool = False
    # Whether to request remote ML attribute classification
    use_ml_attributes: bool = False
    # Attribute multiplier weight: how much to trust ML attrs vs heuristic
    ml_attr_weight: float = 0.7
    # Lighting condition
    lighting: str = "normal"  # "dark" | "normal" | "bright"
    # Minimum crop quality (set by agent based on camera history)
    skip_recognition: bool = False  # True = crop failed quality check


@dataclass
class CropQuality:
    """Assessment of a detection crop's suitability for recognition."""

    area: int = 0                  # Total pixels (h × w)
    aspect_ratio: float = 0.0     # h / w
    blur_score: float = 0.0       # Laplacian variance (higher = sharper)
    skin_ratio: float = -1.0      # Fraction of skin-toned pixels (-1 = not computed)
    head_edge_score: float = -1.0 # Edge density in upper third (-1 = not computed)
    is_valid: bool = True          # Whether the crop passes minimum quality
    reject_reason: str = ""        # Why it was rejected


@dataclass
class MatchVerdict:
    """The agent's decision on whether a recognition match should be accepted."""

    accept: bool = True
    confidence_adjustment: float = 1.0  # Multiplier to apply to the match score
    reason: str = ""


@dataclass
class LastAppearance:
    """Tracks what a recognised person looked like recently (for continuity checks)."""

    camera_id: int = 0
    timestamp: float = 0.0
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    hair_color: Optional[str] = None
    skin_tone: Optional[str] = None


@dataclass
class MLAttributeResult:
    """Result from the remote ML attribute classifier."""

    gender: Optional[str] = None
    gender_conf: float = 0.0
    age_group: Optional[str] = None
    age_group_conf: float = 0.0
    build: Optional[str] = None
    build_conf: float = 0.0


class RecognitionAgent:
    """Intelligent coordinator for the recognition pipeline.

    The agent is the central decision-maker, performing:
    - Detection quality validation (pre-filter)
    - Adaptive threshold planning
    - Multi-signal attribute fusion
    - Post-match coherence verification
    - Appearance continuity tracking

    Usage from EventProcessor::

        quality = agent.assess_crop_quality(crop, "person")
        if not quality.is_valid:
            return None  # Reject this detection

        plan = agent.plan_recognition(camera_id, frame, det)
        # plan.face_threshold, plan.try_face, etc. inform the pipeline
        ml_attrs = await agent.classify_attributes(crop) if plan.use_ml_attributes else None
        multiplier = agent.compute_fused_multiplier(heuristic_attrs, ml_attrs, stored_profile, plan)

        # After getting a match candidate:
        verdict = agent.validate_match(camera_id, match_id, match_name, score, person_attrs)
        if not verdict.accept:
            match_id = None  # Agent rejected the match

        agent.record_outcome(camera_id, "face", success=True)
    """

    def __init__(self):
        self._contexts: dict[int, CameraContext] = {}
        self._ml_attr_available: Optional[bool] = None  # Lazy-checked
        self._ml_attr_check_ts: float = 0.0
        # Track recent appearances for clothing continuity
        # Key: (camera_id, named_object_id) → LastAppearance
        self._last_appearances: dict[tuple[int, int], LastAppearance] = {}

    def _get_context(self, camera_id: int) -> CameraContext:
        if camera_id not in self._contexts:
            self._contexts[camera_id] = CameraContext(camera_id=camera_id)
        ctx = self._contexts[camera_id]
        ctx.maybe_reset_window()
        return ctx

    # ── Scene analysis ──

    def _assess_lighting(self, frame: np.ndarray, camera_id: int) -> str:
        """Assess scene lighting from frame luminance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        ctx = self._get_context(camera_id)
        ctx.luminance_history.append(mean_lum)
        if len(ctx.luminance_history) > _LIGHTING_HISTORY_SIZE:
            ctx.luminance_history = ctx.luminance_history[-_LIGHTING_HISTORY_SIZE:]

        avg = ctx.avg_luminance
        if avg < 50:
            return "dark"
        if avg > 200:
            return "bright"
        return "normal"

    # ── Detection quality validation ──

    def assess_crop_quality(self, crop: np.ndarray, class_name: str) -> CropQuality:
        """Evaluate whether a detection crop is suitable for recognition.

        Rejects crops that are too small, too blurry, or have implausible
        aspect ratios — these are almost always false YOLO detections
        (inanimate objects, noise, reflections).
        """
        quality = CropQuality()
        h, w = crop.shape[:2]
        quality.area = h * w
        quality.aspect_ratio = h / max(w, 1)

        # Blur assessment via Laplacian variance
        if h >= 10 and w >= 10:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            quality.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            quality.blur_score = 0.0

        if class_name == "person":
            min_area = MIN_PERSON_CROP_AREA
            if quality.area < min_area:
                quality.is_valid = False
                quality.reject_reason = f"too_small ({quality.area}px < {min_area}px)"
                return quality

            if quality.aspect_ratio < MIN_PERSON_ASPECT_RATIO:
                quality.is_valid = False
                quality.reject_reason = f"aspect_ratio ({quality.aspect_ratio:.2f} < {MIN_PERSON_ASPECT_RATIO})"
                return quality

            if quality.aspect_ratio > MAX_PERSON_ASPECT_RATIO:
                quality.is_valid = False
                quality.reject_reason = f"aspect_ratio ({quality.aspect_ratio:.2f} > {MAX_PERSON_ASPECT_RATIO})"
                return quality

            if quality.blur_score < MIN_PERSON_BLUR_SCORE:
                quality.is_valid = False
                quality.reject_reason = f"too_blurry (score={quality.blur_score:.1f} < {MIN_PERSON_BLUR_SCORE})"
                return quality

            # ── Skin presence check ──
            # People have skin — cabinets, chairs, and walls don't.
            # Uses HSV ranges covering diverse skin tones.
            try:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                # Range 1: warm skin tones (light to medium)
                mask1 = cv2.inRange(hsv, (0, 25, 60), (25, 255, 255))
                # Range 2: reddish/darker skin tones
                mask2 = cv2.inRange(hsv, (160, 25, 60), (180, 255, 255))
                skin_mask = mask1 | mask2
                quality.skin_ratio = float(np.count_nonzero(skin_mask)) / max(quality.area, 1)
            except Exception:
                quality.skin_ratio = -1.0

            # ── Head region edge density ──
            # People have heads with strong edges in the upper portion;
            # furniture and walls are comparatively flat/uniform.
            try:
                if h >= 30:
                    gray_upper = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    head_region = gray_upper[: h // 3, :]
                    edges = cv2.Canny(head_region, 50, 150)
                    quality.head_edge_score = float(np.count_nonzero(edges)) / max(edges.size, 1)
                else:
                    quality.head_edge_score = 0.0
            except Exception:
                quality.head_edge_score = -1.0

            # Reject crops with NO skin AND no meaningful edges in head region.
            # This catches furniture, walls, and other inanimate objects that
            # Frigate's YOLO falsely classified as "person".
            if quality.skin_ratio >= 0 and quality.head_edge_score >= 0:
                if quality.skin_ratio < 0.01 and quality.head_edge_score < 0.02:
                    quality.is_valid = False
                    quality.reject_reason = (
                        f"not_a_person (skin={quality.skin_ratio:.3f}, "
                        f"head_edges={quality.head_edge_score:.3f})"
                    )
                    return quality

        elif class_name in ("cat", "dog"):
            if quality.area < MIN_PET_CROP_AREA:
                quality.is_valid = False
                quality.reject_reason = f"too_small ({quality.area}px < {MIN_PET_CROP_AREA}px)"
                return quality

        return quality

    # ── Recognition planning ──

    def plan_recognition(
        self,
        camera_id: int,
        frame: np.ndarray,
        det_bbox: tuple[int, int, int, int],
        det_class: str = "person",
    ) -> RecognitionPlan:
        """Produce an adaptive recognition plan based on scene context.

        Called once per detection before the pipeline begins.
        """
        plan = RecognitionPlan()
        ctx = self._get_context(camera_id)

        # Assess current lighting
        plan.lighting = self._assess_lighting(frame, camera_id)

        # ── Threshold adaptation ──

        # Dark scenes: face embeddings are less reliable → widen acceptance
        if plan.lighting == "dark":
            plan.face_threshold = 0.25  # More lenient (base 0.28)
            plan.body_threshold = 0.38  # Slightly more lenient
        elif plan.lighting == "bright":
            plan.face_threshold = 0.30  # Slightly stricter in good light
            plan.body_threshold = 0.43

        # Camera with historically low face detection rate
        # (high angle, back-facing, etc.) → prefer body ReID
        if ctx.face_detect_rate < 0.15 and ctx.face_attempts >= 10:
            plan.prefer_body_first = True
            # Still try face but don't wait as long
            plan.try_face = True
            logger.debug(
                "Camera %d: low face rate (%.0f%%) → body-first mode",
                camera_id, ctx.face_detect_rate * 100,
            )

        # Camera that rarely matches on face → lower threshold to catch more
        if ctx.face_hit_rate < 0.1 and ctx.face_attempts >= 20:
            plan.face_threshold = max(0.22, plan.face_threshold - 0.03)

        # High-activity camera (many detections) → tighten to reduce false positives
        total = ctx.face_attempts + ctx.body_attempts
        if total > 100 and (ctx.face_hit_rate + ctx.body_hit_rate) / 2 > 0.4:
            plan.face_threshold = min(0.33, plan.face_threshold + 0.02)
            plan.body_threshold = min(0.46, plan.body_threshold + 0.02)

        # ── ML attribute model ──
        # Use ML attributes when available and detection is a person
        if det_class == "person" and self._is_ml_attr_available():
            plan.use_ml_attributes = True

        return plan

    # ── Remote ML attribute classification ──

    def _is_ml_attr_available(self) -> bool:
        """Check (lazily, cached 60s) if the remote ML server supports attributes."""
        now = time.monotonic()
        if self._ml_attr_available is not None and now - self._ml_attr_check_ts < 60:
            return self._ml_attr_available
        # Will be set async on first call
        return self._ml_attr_available or False

    async def _check_ml_attr_endpoint(self):
        """Probe the remote ML server for /v1/attributes support."""
        try:
            from services.ml_client import ml_offload_enabled, ml_offload_url, _get_client
            if not ml_offload_enabled:
                self._ml_attr_available = False
                self._ml_attr_check_ts = time.monotonic()
                return
            client = await _get_client()
            resp = await client.get(f"{ml_offload_url}/health", timeout=5.0)
            data = resp.json()
            # Check if 'attributes' model is listed
            models = data.get("models", {})
            self._ml_attr_available = bool(models.get("attributes"))
            self._ml_attr_check_ts = time.monotonic()
            if self._ml_attr_available:
                logger.info("ML attribute classifier available on remote server")
        except Exception:
            self._ml_attr_available = False
            self._ml_attr_check_ts = time.monotonic()

    async def classify_attributes(self, crop: np.ndarray) -> Optional[MLAttributeResult]:
        """Call the remote ML server for multi-task attribute classification.

        The remote endpoint (``/v1/attributes``) runs a specialized model that
        predicts gender, age group, and build from a person crop — far more
        accurate than the heuristic estimator, especially without face data.

        Falls back gracefully if the endpoint isn't available.
        """
        # Lazy-check availability
        if self._ml_attr_available is None:
            await self._check_ml_attr_endpoint()
        if not self._ml_attr_available:
            return None

        try:
            from services.ml_client import ml_offload_url, _get_client, _encode_frame
            client = await _get_client()
            payload = {"image": _encode_frame(crop)}
            resp = await client.post(
                f"{ml_offload_url}/v1/attributes", json=payload, timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

            result = MLAttributeResult()
            if "gender" in data:
                result.gender = data["gender"]
                result.gender_conf = float(data.get("gender_confidence", 0.8))
            if "age_group" in data:
                result.age_group = data["age_group"]
                result.age_group_conf = float(data.get("age_group_confidence", 0.7))
            if "build" in data:
                result.build = data["build"]
                result.build_conf = float(data.get("build_confidence", 0.6))
            return result
        except Exception as e:
            logger.debug("ML attribute classification failed: %s", e)
            return None

    # ── Fused attribute multiplier ──

    def compute_fused_multiplier(
        self,
        heuristic_attrs: PersonAttributes,
        ml_attrs: Optional[MLAttributeResult],
        stored_profile: Optional[dict],
        plan: RecognitionPlan,
    ) -> float:
        """Compute a confidence multiplier by fusing heuristic + ML attributes.

        When the ML attribute model is available, its predictions are blended
        with the heuristic estimator using the plan's ``ml_attr_weight``.
        This produces a more robust multiplier than either source alone.
        """
        # Start with the standard heuristic multiplier
        base_mult = compute_attribute_multiplier(heuristic_attrs, stored_profile)

        if ml_attrs is None or stored_profile is None:
            return base_mult

        # Compute ML-based multiplier
        ml_mult = 1.0

        # Gender from ML model
        stored_gender = stored_profile.get("gender")
        gender_manual = stored_profile.get("_gender_manual", False)
        if stored_gender and ml_attrs.gender and ml_attrs.gender_conf >= 0.6:
            if ml_attrs.gender == stored_gender:
                ml_mult *= 1.08  # ML gender is more reliable → stronger bonus
            else:
                ml_mult *= 0.60 if gender_manual else 0.78

        # Age group from ML model
        stored_age = stored_profile.get("age_group")
        age_manual = stored_profile.get("_age_group_manual", False)
        if stored_age and ml_attrs.age_group and ml_attrs.age_group_conf >= 0.5:
            from services.attribute_estimator import AGE_GROUP_ORDER
            s_idx = AGE_GROUP_ORDER.index(stored_age) if stored_age in AGE_GROUP_ORDER else -1
            d_idx = AGE_GROUP_ORDER.index(ml_attrs.age_group) if ml_attrs.age_group in AGE_GROUP_ORDER else -1
            if s_idx >= 0 and d_idx >= 0:
                gap = abs(s_idx - d_idx)
                if gap == 0:
                    ml_mult *= 1.06
                elif gap >= 3:
                    ml_mult *= 0.40 if age_manual else 0.50  # Extreme mismatch
                elif gap >= 2:
                    ml_mult *= 0.55 if age_manual else 0.65

        # Build from ML model
        stored_build = stored_profile.get("build")
        if stored_build and ml_attrs.build and ml_attrs.build_conf >= 0.5:
            build_order = ["slim", "medium", "large"]
            s_bi = build_order.index(stored_build) if stored_build in build_order else -1
            d_bi = build_order.index(ml_attrs.build) if ml_attrs.build in build_order else -1
            if s_bi >= 0 and d_bi >= 0:
                gap = abs(s_bi - d_bi)
                if gap == 0:
                    ml_mult *= 1.04
                elif gap >= 2:
                    ml_mult *= 0.85

        # Clamp ML multiplier
        ml_mult = max(0.45, min(1.20, ml_mult))

        # Blend: weighted average of heuristic and ML multipliers
        w = plan.ml_attr_weight
        fused = (1.0 - w) * base_mult + w * ml_mult

        # Final clamp — lowered floor to allow strong attribute penalties
        # to actually prevent matches (0.75 was too protective)
        return max(0.55, min(1.20, fused))

    # ── Post-match validation ──

    def validate_match(
        self,
        camera_id: int,
        named_object_id: int,
        named_object_name: str,
        raw_score: float,
        adjusted_score: float,
        person_attrs: PersonAttributes,
        stored_profile: Optional[dict],
        match_method: str = "face",
        det_class: str = "person",
        det_confidence: float = 1.0,
    ) -> MatchVerdict:
        """Verify that a recognition match is coherent with recent observations.

        **These are real visual checks** — gender, hair, and skin tone are
        derived from InsightFace face analysis and pixel-level image processing,
        not just embedding distance maths.

        Checks:
        1. Gender veto — if the stored profile has an established gender
           (manual or learned with high confidence) that conflicts with
           the detected gender from face analysis, reject outright.
           Example: "woman with long hair" ≠ Philip (male).
        2. Clothing continuity — if we saw this person seconds ago on the
           same camera, their clothing shouldn't have drastically changed.
        3. Hair colour veto — stored profile says "black hair" but the crop
           shows blonde? Hard reject for well-established profiles.
        4. Skin tone veto — stored profile says "light" but crop shows "dark"?
           Hard reject for well-established profiles.
        5. YOLO confidence check — when the detector itself is uncertain
           (< 55%), require stronger recognition to compensate.
        6. Score quality — marginal body-only matches get penalised.

        Returns a MatchVerdict with accept=True/False and a reason.
        """
        verdict = MatchVerdict()

        # ── CHECK 1: Hard gender veto ──
        # Fires when:
        #   (a) Profile has manually-confirmed gender, OR
        #   (b) Profile has learned gender with high confidence (≥0.80, ≥10 samples)
        # AND the detected gender disagrees with ≥70% confidence.
        # This is a genuine visual check — gender comes from InsightFace's
        # face analysis model or the remote ML attribute classifier, both of
        # which actually look at the image pixels.
        if stored_profile and person_attrs.gender and person_attrs.gender_conf >= 0.70:
            stored_gender = stored_profile.get("gender")
            gender_manual = stored_profile.get("_gender_manual", False)
            gender_confidence = stored_profile.get("gender_confidence", 0)
            gender_samples = stored_profile.get("gender_samples", 0)

            if stored_gender and stored_gender != person_attrs.gender:
                # Hard veto for manual profiles — always
                # Hard veto for learned profiles — only when well-established
                is_established = (
                    gender_manual
                    or (gender_confidence >= 0.80 and gender_samples >= 10)
                )
                if is_established:
                    verdict.accept = False
                    source = "manual" if gender_manual else f"learned, conf={gender_confidence:.2f}, n={gender_samples}"
                    verdict.reason = (
                        f"gender_veto (stored={stored_gender}[{source}], "
                        f"detected={person_attrs.gender} conf={person_attrs.gender_conf:.2f})"
                    )
                    logger.info(
                        "Match VETOED by gender: %s on camera %d — %s",
                        named_object_name, camera_id, verdict.reason,
                    )
                    return verdict

        # ── CHECK 2: Hard age-group veto ──
        # A child cannot be an adult or older, and vice versa.  This is a
        # fundamental physical impossibility that no embedding score should
        # override — face embeddings between mother and daughter can be very
        # similar, but a 170cm adult woman is not a 120cm child.
        if stored_profile and person_attrs.age_group:
            from services.attribute_estimator import AGE_GROUP_ORDER
            stored_age_group = stored_profile.get("age_group")
            age_manual = stored_profile.get("_age_group_manual", False)
            age_samples = stored_profile.get("age_group_samples", 0)
            age_conf = stored_profile.get("age_group_confidence", 0)
            is_age_established = (
                age_manual
                or (age_conf >= 0.70 and age_samples >= 5)
            )
            if stored_age_group and is_age_established and stored_age_group in AGE_GROUP_ORDER:
                s_idx = AGE_GROUP_ORDER.index(stored_age_group)
                d_idx = AGE_GROUP_ORDER.index(person_attrs.age_group) if person_attrs.age_group in AGE_GROUP_ORDER else -1
                if d_idx >= 0:
                    age_gap = abs(s_idx - d_idx)
                    # Gap ≥ 2 = hard veto (child ↔ adult, child ↔ middle_aged, etc.)
                    if age_gap >= 2:
                        source = "manual" if age_manual else f"learned, conf={age_conf:.2f}, n={age_samples}"
                        verdict.accept = False
                        verdict.reason = (
                            f"age_veto (stored={stored_age_group}[{source}], "
                            f"detected={person_attrs.age_group}, gap={age_gap})"
                        )
                        logger.info(
                            "Match VETOED by age group: %s on camera %d — %s",
                            named_object_name, camera_id, verdict.reason,
                        )
                        return verdict

        # ── CHECK 3: Height ratio veto ──
        # If the stored profile has a well-established height ratio and
        # the detected height differs by more than 50%, reject.  Children
        # are much shorter than adults — this catches size mismatches.
        if stored_profile and person_attrs.height_ratio > 0:
            stored_hr = stored_profile.get("height_ratio", 0)
            height_n = stored_profile.get("_height_n", 0)
            if stored_hr > 0 and height_n >= 5:
                hr_diff = abs(person_attrs.height_ratio - stored_hr) / max(stored_hr, 0.01)
                if hr_diff > 0.50:
                    verdict.accept = False
                    verdict.reason = (
                        f"height_veto (stored={stored_hr:.3f} n={height_n}, "
                        f"detected={person_attrs.height_ratio:.3f}, diff={hr_diff:.0%})"
                    )
                    logger.info(
                        "Match VETOED by height ratio: %s on camera %d — %s",
                        named_object_name, camera_id, verdict.reason,
                    )
                    return verdict

        # Check clothing continuity with recent appearance
        key = (camera_id, named_object_id)
        last = self._last_appearances.get(key)
        now = time.monotonic()

        if last and (now - last.timestamp) < CLOTHING_MEMORY_TTL:
            clothing_mismatches = 0
            total_checks = 0

            # Upper clothing colour — should be consistent within 5 minutes
            if last.upper_color and person_attrs.upper_color and last.upper_color != "unknown" and person_attrs.upper_color != "unknown":
                total_checks += 1
                if last.upper_color != person_attrs.upper_color:
                    clothing_mismatches += 1

            # Lower clothing colour
            if last.lower_color and person_attrs.lower_color and last.lower_color != "unknown" and person_attrs.lower_color != "unknown":
                total_checks += 1
                if last.lower_color != person_attrs.lower_color:
                    clothing_mismatches += 1

            # If BOTH upper and lower mismatch, and the match is not very strong,
            # this is likely a different person inheriting the identity
            if clothing_mismatches >= 2 and total_checks >= 2:
                age = now - last.timestamp
                if age < 60:
                    # Within 1 minute — very suspicious
                    if adjusted_score < 0.45:
                        verdict.accept = False
                        verdict.reason = f"clothing_mismatch (upper: {last.upper_color}→{person_attrs.upper_color}, lower: {last.lower_color}→{person_attrs.lower_color}, {age:.0f}s ago)"
                        logger.info(
                            "Match rejected by clothing continuity: %s on camera %d — %s",
                            named_object_name, camera_id, verdict.reason,
                        )
                        return verdict
                    else:
                        # Strong embedding match — penalise but don't reject
                        verdict.confidence_adjustment *= 0.88

            # Hair colour continuity with recent observation
            if last.hair_color and person_attrs.hair_color:
                if last.hair_color != person_attrs.hair_color and age < 120:
                    verdict.confidence_adjustment *= 0.92

        # Check hair colour against stored profile (stable biometric)
        if stored_profile and person_attrs.hair_color:
            stored_hair = stored_profile.get("hair_color")
            hair_samples = stored_profile.get("hair_color_samples", 0)
            hair_manual = stored_profile.get("_hair_color_manual", False)
            if stored_hair and hair_samples >= 5 and person_attrs.hair_color != stored_hair:
                _hair_order = ["black", "brown", "red", "blonde", "grey", "white"]
                s_hi = _hair_order.index(stored_hair) if stored_hair in _hair_order else -1
                d_hi = _hair_order.index(person_attrs.hair_color) if person_attrs.hair_color in _hair_order else -1
                if s_hi >= 0 and d_hi >= 0 and abs(s_hi - d_hi) >= 3:
                    # Major hair colour mismatch (e.g. black vs blonde)
                    # Hard veto for well-established profiles (manual or ≥20 samples)
                    if hair_manual or hair_samples >= 20:
                        verdict.accept = False
                        source = "manual" if hair_manual else f"learned, n={hair_samples}"
                        verdict.reason = f"hair_veto (stored={stored_hair}[{source}], detected={person_attrs.hair_color})"
                        logger.info(
                            "Match VETOED by hair colour: %s — %s",
                            named_object_name, verdict.reason,
                        )
                        return verdict
                    elif adjusted_score < 0.40:
                        verdict.accept = False
                        verdict.reason = f"hair_mismatch (stored={stored_hair}, detected={person_attrs.hair_color})"
                        logger.info(
                            "Match rejected by hair colour: %s — %s",
                            named_object_name, verdict.reason,
                        )
                        return verdict
                    verdict.confidence_adjustment *= 0.85

        # Check skin tone against stored profile
        if stored_profile and person_attrs.skin_tone:
            stored_skin = stored_profile.get("skin_tone")
            skin_samples = stored_profile.get("skin_tone_samples", 0)
            skin_manual = stored_profile.get("_skin_tone_manual", False)
            if stored_skin and skin_samples >= 5:
                _tone_order = ["light", "medium", "dark"]
                s_ti = _tone_order.index(stored_skin) if stored_skin in _tone_order else -1
                d_ti = _tone_order.index(person_attrs.skin_tone) if person_attrs.skin_tone in _tone_order else -1
                if s_ti >= 0 and d_ti >= 0 and abs(s_ti - d_ti) >= 2:
                    # Light vs dark is a definitive mismatch
                    # Hard veto for well-established profiles
                    if skin_manual or skin_samples >= 20:
                        verdict.accept = False
                        source = "manual" if skin_manual else f"learned, n={skin_samples}"
                        verdict.reason = f"skin_tone_veto (stored={stored_skin}[{source}], detected={person_attrs.skin_tone})"
                        logger.info(
                            "Match VETOED by skin tone: %s — %s",
                            named_object_name, verdict.reason,
                        )
                        return verdict
                    elif adjusted_score < 0.42:
                        verdict.accept = False
                        verdict.reason = f"skin_tone_mismatch (stored={stored_skin}, detected={person_attrs.skin_tone})"
                        logger.info(
                            "Match rejected by skin tone: %s — %s",
                            named_object_name, verdict.reason,
                        )
                        return verdict
                    verdict.confidence_adjustment *= 0.82

        # Body-only matches need higher quality — they're more prone to false positives
        if match_method == "body" and raw_score < 0.50 and adjusted_score < 0.48:
            verdict.confidence_adjustment *= 0.90

        # ── CHECK: YOLO confidence weighting ──
        # When YOLO is barely confident this is a person (< 0.55), the
        # recognition match needs to be significantly stronger to compensate.
        # Low YOLO confidence means the detection itself is uncertain —
        # accepting a borderline recognition on a borderline detection is
        # how misidentifications happen.
        if det_confidence < 0.55 and adjusted_score < 0.50:
            verdict.accept = False
            verdict.reason = (
                f"low_yolo_confidence_veto (YOLO={det_confidence:.2f}, "
                f"recognition={adjusted_score:.3f} — both too marginal)"
            )
            logger.info(
                "Match rejected: %s on camera %d — %s",
                named_object_name, camera_id, verdict.reason,
            )
            return verdict
        elif det_confidence < 0.55:
            # YOLO uncertain but recognition strong — penalise
            verdict.confidence_adjustment *= 0.90

        return verdict

    # ── Non-person match validation ──

    async def validate_pet_match(
        self,
        det_class: str,
        matched_name: str,
        matched_id: int,
        cosine_score: float,
        matched_category: str,
        crop: np.ndarray,
        det_confidence: float = 1.0,
    ) -> MatchVerdict:
        """Validate a non-person (pet/vehicle) CNN match with visual sanity checks.

        This is the agent *actually looking* at the detection to decide whether
        the match makes sense — not just trusting the cosine similarity.

        Checks:
        1. **Category-class consistency** — YOLO class must be compatible with
           the matched named object's category.  A "cat" detection matched to
           a "vehicle" named object is rejected unconditionally.
        2. **YOLO re-verification** — re-run YOLO on the tight crop to confirm
           the expected animal class is actually present.  A bag that YOLO
           misclassified as "cat" will fail this re-check because the crop
           itself doesn't look like a cat.
        3. **Aspect-ratio plausibility** — cats/dogs are roughly square-ish,
           vehicles are wide.  A crop with a person-like tall aspect ratio
           should not be accepted as a pet and vice-versa.
        4. **Minimum quality gate** — very small or blurry crops get a score
           penalty because CNN embeddings from poor crops are unreliable.
        5. **Minimum YOLO confidence gate** — require ≥0.50 confidence for
           pet class detections to prevent low-confidence misclassifications.
        """
        verdict = MatchVerdict()

        # ── Class → category consistency (hard veto) ──
        _CLASS_TO_CATEGORY = {
            "cat": "pet", "dog": "pet",
            "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
            "motorcycle": "vehicle", "bicycle": "vehicle",
            "boat": "vehicle", "train": "vehicle", "airplane": "vehicle",
            "person": "person",
        }
        expected_cat = _CLASS_TO_CATEGORY.get(det_class)
        if expected_cat and matched_category and expected_cat != matched_category:
            verdict.accept = False
            verdict.reason = (
                f"category_mismatch (YOLO={det_class}→{expected_cat}, "
                f"matched={matched_name}→{matched_category})"
            )
            logger.warning(
                "Pet/vehicle match VETOED: %s — %s",
                matched_name, verdict.reason,
            )
            return verdict

        # ── Minimum YOLO confidence gate for pets ──
        if det_class in ("cat", "dog") and det_confidence < 0.50:
            verdict.accept = False
            verdict.reason = (
                f"low_pet_yolo_conf (YOLO={det_class} conf={det_confidence:.2f} < 0.50)"
            )
            logger.info("Pet match rejected: %s — %s", matched_name, verdict.reason)
            return verdict

        # ── YOLO re-verification on crop ──
        # Re-run YOLO on the tight crop itself.  A real cat/dog will be
        # detected again in its own crop; a bag or shoe won't.
        if det_class in ("cat", "dog"):
            try:
                from services.object_detector import object_detector
                re_dets = await object_detector.detect(
                    crop,
                    confidence_threshold=0.35,
                    target_classes=["cat", "dog", "bird", "teddy bear"],
                )
                found_animal = any(
                    d.class_name in ("cat", "dog") and d.confidence >= 0.40
                    for d in re_dets
                )
                if not found_animal:
                    verdict.accept = False
                    re_classes = [(d.class_name, f"{d.confidence:.2f}") for d in re_dets] if re_dets else []
                    verdict.reason = (
                        f"yolo_reverify_fail (re-ran YOLO on crop, no cat/dog found. "
                        f"Got: {re_classes})"
                    )
                    logger.info(
                        "Pet match VETOED by re-verification: %s — %s",
                        matched_name, verdict.reason,
                    )
                    return verdict
                else:
                    logger.debug(
                        "Pet re-verification passed for %s: found %s in crop",
                        matched_name, [(d.class_name, f"{d.confidence:.2f}") for d in re_dets],
                    )
            except Exception as e:
                # Re-verification failed — log but don't block the match
                logger.debug("Pet re-verification error (non-fatal): %s", e)

        # ── Segmentation mask coverage check ──
        # If the detection has a seg mask, check that the object fills a
        # reasonable portion of its bounding box.  Real animals fill their
        # bbox; bags and random objects often don't.
        if det_class in ("cat", "dog") and hasattr(crop, '_mask') and crop._mask is not None:
            mask = crop._mask
            mask_coverage = float(np.count_nonzero(mask)) / max(mask.size, 1)
            if mask_coverage < 0.20:
                verdict.accept = False
                verdict.reason = f"low_mask_coverage ({mask_coverage:.0%} < 20%)"
                logger.info("Pet match rejected: %s — %s", matched_name, verdict.reason)
                return verdict

        # ── Aspect-ratio plausibility ──
        h, w = crop.shape[:2]
        aspect = h / max(w, 1)

        if det_class in ("cat", "dog"):
            # Pets are roughly square (0.4 – 2.5).
            # A very tall narrow crop (aspect > 3) looks like a person, not a pet.
            if aspect > 3.0:
                verdict.accept = False
                verdict.reason = f"aspect_implausible_for_pet (h/w={aspect:.2f} > 3.0, looks like a person crop)"
                logger.info("Pet match rejected: %s — %s", matched_name, verdict.reason)
                return verdict
        elif det_class in ("car", "truck", "bus"):
            # Vehicles are wide (aspect < ~1.5 normally).
            # A very tall crop is implausible.
            if aspect > 2.5:
                verdict.confidence_adjustment *= 0.80

        # ── Crop quality penalty ──
        area = h * w
        if area < MIN_PET_CROP_AREA:
            # Very small crop — CNN embedding is unreliable
            if cosine_score < 0.55:
                verdict.accept = False
                verdict.reason = f"low_quality_pet_crop (area={area}, score={cosine_score:.3f})"
                logger.info("Pet match rejected: %s — %s", matched_name, verdict.reason)
                return verdict
            verdict.confidence_adjustment *= 0.85

        return verdict

    def record_appearance(
        self,
        camera_id: int,
        named_object_id: int,
        person_attrs: PersonAttributes,
    ):
        """Record what a recognised person looks like right now (for continuity checks)."""
        key = (camera_id, named_object_id)
        self._last_appearances[key] = LastAppearance(
            camera_id=camera_id,
            timestamp=time.monotonic(),
            upper_color=person_attrs.upper_color,
            lower_color=person_attrs.lower_color,
            hair_color=person_attrs.hair_color,
            skin_tone=person_attrs.skin_tone,
        )

        # Prune stale entries (avoid memory leak)
        now = time.monotonic()
        if len(self._last_appearances) > 200:
            stale = [k for k, v in self._last_appearances.items()
                     if now - v.timestamp > CLOTHING_MEMORY_TTL * 2]
            for k in stale:
                del self._last_appearances[k]

    # ── Outcome recording ──

    def record_outcome(
        self,
        camera_id: int,
        method: str,
        success: bool,
        face_detected: bool = False,
    ):
        """Record a recognition attempt outcome to update camera context.

        Parameters
        ----------
        camera_id : which camera
        method : "face" or "body"
        success : whether the match succeeded
        face_detected : whether a face was found at all (for face detect rate)
        """
        ctx = self._get_context(camera_id)

        if method == "face":
            ctx.face_attempts += 1
            if success:
                ctx.face_hits += 1
            # Update face detection rate (exponential moving average)
            alpha = 0.1
            ctx.face_detect_rate = (1 - alpha) * ctx.face_detect_rate + alpha * (1.0 if face_detected else 0.0)
        elif method == "body":
            ctx.body_attempts += 1
            if success:
                ctx.body_hits += 1

    # ── ML-enhanced attribute learning ──

    async def enhance_attributes(
        self,
        person_attrs: PersonAttributes,
        crop: np.ndarray,
        plan: RecognitionPlan,
    ) -> PersonAttributes:
        """Optionally enrich heuristic attributes with ML predictions.

        When the ML model is available and the plan requests it, overlays
        ML-predicted gender/age/build onto the heuristic attributes (only
        when the ML model is more confident).
        """
        if not plan.use_ml_attributes:
            return person_attrs

        ml_result = await self.classify_attributes(crop)
        if ml_result is None:
            return person_attrs

        # Overlay ML predictions when they're more confident
        if ml_result.gender and ml_result.gender_conf > max(person_attrs.gender_conf, 0.5):
            person_attrs.gender = ml_result.gender
            person_attrs.gender_conf = ml_result.gender_conf

        if ml_result.age_group and ml_result.age_group_conf > 0.6:
            if person_attrs.age_group is None:
                person_attrs.age_group = ml_result.age_group

        if ml_result.build and ml_result.build_conf > 0.6:
            if person_attrs.build is None or ml_result.build_conf > 0.75:
                person_attrs.build = ml_result.build

        return person_attrs

    # ── Diagnostics ──

    def get_camera_stats(self) -> dict:
        """Return per-camera recognition statistics for diagnostics."""
        stats = {}
        for cam_id, ctx in self._contexts.items():
            stats[cam_id] = {
                "avg_luminance": round(ctx.avg_luminance, 1),
                "face_detect_rate": round(ctx.face_detect_rate, 2),
                "face_hit_rate": round(ctx.face_hit_rate, 2),
                "body_hit_rate": round(ctx.body_hit_rate, 2),
                "face_attempts": ctx.face_attempts,
                "body_attempts": ctx.body_attempts,
                "active_appearances": sum(
                    1 for k in self._last_appearances
                    if k[0] == cam_id
                ),
            }
        return stats


# Module-level singleton
recognition_agent = RecognitionAgent()
