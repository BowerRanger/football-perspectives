raise ImportError("prepare_shots stage has been removed")


def _get_video_fps_and_frames(clip_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            return 0.0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return fps, frame_count
    finally:
        cap.release()


# Minimum mean Sobel gradient to trust gradient normalisation.  Below this the
# frame is near-uniform (e.g. black cut, out-of-focus) and we fall back to
# comparing raw flow magnitudes.
_MIN_GRADIENT: float = 0.1


def _sample_normalized_motion(
    clip_path: Path,
    n_samples: int,
) -> tuple[float, float]:
    """Sample optical-flow and Sobel-gradient magnitudes from a clip.

    Returns ``(mean_flow, mean_gradient)`` averaged across *n_samples* evenly-
    spaced consecutive frame pairs.

    *mean_gradient* is the mean per-pixel Sobel-RMS gradient of the first frame
    in each sampled pair.  It scales roughly proportionally with the camera's
    zoom / field-of-view, making ``mean_flow / mean_gradient`` a zoom-invariant
    measure of the temporal rate of motion.
    """
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            return 0.0, 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames < 2:
            return 0.0, 0.0

        effective_samples = max(1, min(n_samples, total_frames // 2))
        max_start_idx = max(0, total_frames - 2)
        if effective_samples == 1:
            sample_indices = [0]
        else:
            sample_indices = np.linspace(0, max_start_idx, num=effective_samples, dtype=int).tolist()

        flow_magnitudes: list[float] = []
        gradient_magnitudes: list[float] = []
        for start_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx))
            ok_a, frame_a = cap.read()
            ok_b, frame_b = cap.read()
            if not ok_a or not ok_b:
                continue

            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

            # Sobel gradient measures spatial detail density, which scales with zoom.
            grad_x = cv2.Sobel(gray_a, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_a, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitudes.append(float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))))

            corners = cv2.goodFeaturesToTrack(
                gray_a,
                maxCorners=250,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
            )
            if corners is None or len(corners) == 0:
                continue

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_a,
                gray_b,
                corners,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_points is None or status is None:
                continue

            valid_mask = status.reshape(-1) == 1
            if not np.any(valid_mask):
                continue

            tracked_a = corners.reshape(-1, 2)[valid_mask]
            tracked_b = next_points.reshape(-1, 2)[valid_mask]
            displacement = np.linalg.norm(tracked_b - tracked_a, axis=1)
            if displacement.size == 0:
                continue
            flow_magnitudes.append(float(np.mean(displacement)))

        mean_flow = float(np.mean(np.array(flow_magnitudes, dtype=float))) if flow_magnitudes else 0.0
        mean_gradient = float(np.mean(np.array(gradient_magnitudes, dtype=float))) if gradient_magnitudes else 0.0
        return mean_flow, mean_gradient
    finally:
        cap.release()


def _estimate_speed_factor(
    ref_clip: Path,
    shot_clip: Path,
    n_samples: int,
    min_flow_magnitude: float,
) -> float:
    """Estimate the multiplicative speed factor needed to retime *shot_clip* to
    match the temporal pace of *ref_clip*.

    Returns a value > 1.0 when the shot is slower than the reference (slow-motion
    replay) and < 1.0 when it is faster.  Optical flow is normalised by the frame
    Sobel gradient so that camera zoom / field-of-view differences cancel out.
    """
    ref_flow, ref_grad = _sample_normalized_motion(ref_clip, n_samples)
    shot_flow, shot_grad = _sample_normalized_motion(shot_clip, n_samples)

    # Static scene guard: negligible motion means we can't estimate speed reliably.
    if shot_flow < min_flow_magnitude:
        return 1.0
    if shot_flow <= 1e-6 or ref_flow <= 1e-6:
        return 1.0

    if ref_grad >= _MIN_GRADIENT and shot_grad >= _MIN_GRADIENT:
        # Gradient-normalised comparison removes zoom/FOV confound.
        ref_norm = ref_flow / ref_grad
        shot_norm = shot_flow / shot_grad
        speed_factor = ref_norm / shot_norm
        logging.info(
            "[prepare_shots] speed factor (normalised): "
            "ref_flow=%.2f ref_grad=%.2f shot_flow=%.2f shot_grad=%.2f → %.3f",
            ref_flow, ref_grad, shot_flow, shot_grad, speed_factor,
        )
    else:
        # Near-uniform frames — fall back to raw flow ratio.
        speed_factor = ref_flow / shot_flow
        logging.debug(
            "[prepare_shots] speed factor (raw flow fallback): "
            "ref_flow=%.2f shot_flow=%.2f → %.3f",
            ref_flow, shot_flow, speed_factor,
        )

    return max(0.25, min(4.0, speed_factor))


def _is_reference_shot(shot: Shot) -> bool:
    if shot.id.endswith("01"):
        return True
    clip_stem = Path(shot.clip_file).stem
    return clip_stem.endswith("01")


def _select_reference_shot_index(shots: list[Shot]) -> int:
    for idx, shot in enumerate(shots):
        if _is_reference_shot(shot):
            return idx
    return 0


class ShotPrepStage(BaseStage):
    name = "prepare_shots"

    def is_complete(self) -> bool:
        return (self.output_dir / "prepare_shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        shots_dir = self.output_dir / "shots"
        prepare_dir = self.output_dir / "prepare_shots"
        prepare_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("prepare_shots", {})
        remove_duplicate_frames = bool(cfg.get("remove_duplicate_frames", True))
        n_samples = int(cfg.get("speed_detection_samples", 10))
        speed_factor_threshold = float(cfg.get("speed_factor_threshold", 0.15))
        min_flow_magnitude = float(cfg.get("min_flow_magnitude", 0.5))
        normalise_speed = bool(cfg.get("normalise_speed", True))

        if n_samples <= 0:
            raise ValueError("prepare_shots.speed_detection_samples must be > 0")
        if speed_factor_threshold < 0:
            raise ValueError("prepare_shots.speed_factor_threshold must be >= 0")
        if min_flow_magnitude < 0:
            raise ValueError("prepare_shots.min_flow_magnitude must be >= 0")

        # Load manifest without persisting to shots/ — treat shots/ as read-only input.
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=False)
        prepared_manifest_path = prepare_dir / "shots_manifest.json"

        if not manifest.shots:
            manifest.save(prepared_manifest_path)
            return

        fps = float(manifest.fps)
        reference_idx = _select_reference_shot_index(manifest.shots)

        # ------------------------------------------------------------------
        # Pass 1 — copy every source clip into prepare_shots/, record the
        # original frame count (before dedup), then deduplicate the copy.
        # Source clips in shots/ are never modified.
        # ------------------------------------------------------------------
        dst_clips: list[Path | None] = []
        for shot in manifest.shots:
            src = self.output_dir / shot.clip_file
            if not src.exists():
                logging.warning("[prepare_shots] missing source clip for %s: %s", shot.id, src)
                dst_clips.append(None)
                continue
            dst = prepare_dir / src.name
            shutil.copy2(src, dst)
            if remove_duplicate_frames:
                try:
                    dropped = deduplicate_clip(dst, fps)
                    if dropped:
                        logging.info(
                            "[prepare_shots] %s: removed %d duplicate frame(s)",
                            shot.id,
                            dropped,
                        )
                except subprocess.CalledProcessError as exc:
                    logging.warning(
                        "[prepare_shots] %s: deduplicate failed, keeping copy as-is: %s",
                        shot.id,
                        exc,
                    )
            dst_clips.append(dst)

        # Reference clip is now in prepare_shots/ and already deduplicated.
        ref_clip_path = dst_clips[reference_idx]

        # ------------------------------------------------------------------
        # Pass 2 — assign speed factors and retime non-reference copies.
        # speed_factor records the optical-flow tempo adjustment only.
        # Dedup (freeze-frame removal) is a separate pre-processing step
        # whose frame-count change is intentionally invisible in this field.
        # ------------------------------------------------------------------
        updated_shots: list[Shot] = []
        for idx, (shot, dst_clip) in enumerate(zip(manifest.shots, dst_clips)):
            if dst_clip is None or not dst_clip.exists():
                updated_shots.append(replace(shot))
                continue

            current = replace(shot)
            # Redirect clip_file to the copy inside prepare_shots/.
            current.clip_file = str(dst_clip.relative_to(self.output_dir))

            _, post_dedup_fc = _get_video_fps_and_frames(dst_clip)
            if post_dedup_fc > 0:
                current.end_frame = current.start_frame + post_dedup_fc - 1
                current.end_time = current.start_time + (post_dedup_fc / fps)

            if idx == reference_idx:
                # Reference is the tempo baseline — always 1.0.
                current.speed_factor = 1.0
            else:
                do_retime = (
                    normalise_speed
                    and ref_clip_path is not None
                    and ref_clip_path.exists()
                )
                if do_retime:
                    speed_factor = _estimate_speed_factor(
                        ref_clip=ref_clip_path,
                        shot_clip=dst_clip,
                        n_samples=n_samples,
                        min_flow_magnitude=min_flow_magnitude,
                    )
                    if abs(speed_factor - 1.0) > speed_factor_threshold:
                        try:
                            retime_clip(path=dst_clip, speed_factor=speed_factor, fps=fps)
                            _, retimed_fc = _get_video_fps_and_frames(dst_clip)
                            if retimed_fc > 0:
                                current.end_frame = current.start_frame + retimed_fc - 1
                                current.end_time = current.start_time + (retimed_fc / fps)
                            logging.info(
                                "[prepare_shots] %s: applied speed factor %.3f",
                                shot.id,
                                speed_factor,
                            )
                            current.speed_factor = speed_factor
                        except subprocess.CalledProcessError as exc:
                            logging.warning(
                                "[prepare_shots] %s: retime failed, keeping copy as-is: %s",
                                shot.id,
                                exc,
                            )
                            current.speed_factor = 1.0
                    else:
                        current.speed_factor = 1.0
                else:
                    current.speed_factor = 1.0

            updated_shots.append(current)

        manifest.shots = updated_shots
        manifest.total_frames = sum(
            _get_video_fps_and_frames(self.output_dir / shot.clip_file)[1]
            for shot in updated_shots
            if (self.output_dir / shot.clip_file).exists()
        )
        manifest.save(prepared_manifest_path)
