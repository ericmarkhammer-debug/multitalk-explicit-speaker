# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
MODEL_CACHE = "weights"
BASE_URL = f"https://weights.replicate.delivery/default/multitalk/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import os
import subprocess
import time
import json
import math
import tempfile
import logging
import sys
import warnings
import shutil
from typing import Any, Dict, List, Tuple
from datetime import datetime
from types import SimpleNamespace
from cog import BasePredictor, File, Input, Path as CogPath

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random
import soundfile as sf
from PIL import Image

# MultiTalk (`wan`) is imported lazily in setup()/predict() so Cog schema validation
# can import this module on builders without an NVIDIA driver.
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
import librosa
import pyloudnorm as pyln
from einops import rearrange

logger = logging.getLogger(__name__)


def parse_bbox_input(raw: Any) -> List[float] | None:
    """Parse bbox JSON into [row_min, col_min, row_max, col_max] (see README / Cog Input docs)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid bbox JSON (expected 4 numbers [row_min,col_min,row_max,col_max]): {raw!r}"
            ) from e
    elif isinstance(raw, (list, tuple)):
        parsed = raw
    else:
        raise ValueError(
            f"bbox must be a JSON string or sequence of 4 numbers, got {type(raw).__name__}"
        )
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(
            f"bbox must have exactly 4 values [row_min, col_min, row_max, col_max], got: {parsed!r}"
        )
    try:
        coords = [float(x) for x in parsed]
    except (TypeError, ValueError) as e:
        raise ValueError(f"bbox values must be numeric, got: {parsed!r}") from e
    if not all(math.isfinite(x) for x in coords):
        raise ValueError(f"bbox values must be finite (no nan or inf), got: {coords}")
    row_min, col_min, row_max, col_max = coords
    if row_min >= row_max or col_min >= col_max:
        raise ValueError(
            f"bbox requires row_min < row_max and col_min < col_max, got {coords} "
            f"(wan/multitalk.py uses human_mask[row_min:row_max, col_min:col_max])"
        )
    return coords


def validate_bboxes_against_cond_image(
    person1_bbox: List[float],
    person2_bbox: List[float],
    cond_image_path: str,
) -> Tuple[List[float], List[float], int, int]:
    """
    Clamp bboxes to the cond image size. Values are [row_min, col_min, row_max, col_max], matching wan/multitalk.py:
    human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] on shape [src_h, src_w].
    Returns adjusted boxes and PIL (width, height).
    """
    with Image.open(cond_image_path).convert("RGB") as im:
        img_w, img_h = im.size

    out: List[List[float]] = []
    for label, box in (("person1_bbox", person1_bbox), ("person2_bbox", person2_bbox)):
        row_min, col_min, row_max, col_max = box
        orig = (row_min, col_min, row_max, col_max)
        rx0 = min(max(0.0, float(row_min)), float(img_h))
        rx1 = min(max(0.0, float(row_max)), float(img_h))
        cy0 = min(max(0.0, float(col_min)), float(img_w))
        cy1 = min(max(0.0, float(col_max)), float(img_w))
        if rx0 >= rx1 or cy0 >= cy1:
            raise ValueError(
                f"{label}: bbox invalid or empty after clamping to PIL image width={img_w} height={img_h} "
                f"(original {list(orig)}). Rows [row_min,row_max) must satisfy 0 <= row_min < row_max <= {img_h}; "
                f"cols [col_min,col_max) must satisfy 0 <= col_min < col_max <= {img_w} "
                f"(same as wan/multitalk.py mask indexing)."
            )
        if (rx0, cy0, rx1, cy1) != orig:
            logger.info(
                "%s: clamped bbox from %s to [%s, %s, %s, %s] (PIL WxH=%dx%d)",
                label,
                list(orig),
                rx0,
                cy0,
                rx1,
                cy1,
                img_w,
                img_h,
            )
        out.append([rx0, cy0, rx1, cy1])
    return out[0], out[1], img_w, img_h


def build_bbox_payload(
    person1_bbox: List[float], person2_bbox: List[float]
) -> Dict[str, List[float]]:
    """person1 then person2; each value list is [row_min, col_min, row_max, col_max] for wan/multitalk.py."""
    return {"person1": list(person1_bbox), "person2": list(person2_bbox)}


def resolve_inactive_speaker_mode(
    inactive_speaker_mode: str, second_audio: File | None
) -> str:
    if inactive_speaker_mode == "auto":
        return "second_audio" if second_audio is not None else "none"
    return inactive_speaker_mode


def resolve_multitalk_audio_assignment(
    inactive_eff: str, active_speaker: str | None
) -> Tuple[bool, Dict[str, str | None]]:
    """
    Returns (use_two_audio_files, assign) where assign maps person slots to
    'first' | 'second' | None (which file drives that slot; None = silent / no file).
    """
    if inactive_eff == "second_audio":
        return True, {"person1": "first", "person2": "second"}
    if inactive_eff == "none":
        if active_speaker == "person1":
            return False, {"person1": "first", "person2": None}
        if active_speaker == "person2":
            return False, {"person1": None, "person2": "first"}
        raise ValueError(
            "active_speaker (person1 or person2) is required when inactive_speaker_mode is none."
        )
    raise ValueError(
        f"resolve_multitalk_audio_assignment: unexpected inactive_eff={inactive_eff!r} "
        f"active_speaker={active_speaker!r}"
    )


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def extract_audio_from_video(filename, sample_rate=16000):
    """Extract audio from video file with robust error handling"""
    raw_audio_path = f"{os.path.splitext(os.path.basename(filename))[0]}.wav"
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", str(filename), "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "2", raw_audio_path,
    ]
    subprocess.run(ffmpeg_command, check=True, capture_output=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)
    return human_speech_array


def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    human_speech_array1 = audio_prepare_single(left_path)
    human_speech_array2 = audio_prepare_single(right_path)

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    """Extract audio embeddings optimized for GPU processing"""
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # Extract audio features
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # Generate embeddings on appropriate device
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("⚠️ Failed to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    # Keep on CPU for compatibility with downstream processing
    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        model_files = [
            "MeiGen-MultiTalk.tar",
            "Wan2.1-I2V-14B-480P.tar",
            "chinese-wav2vec2-base.tar"
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
                
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
        
        # Model paths
        self.ckpt_dir = "weights/Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = "weights/chinese-wav2vec2-base"
        self.multitalk_dir = "weights/MeiGen-MultiTalk"
        
        # Initialize device for single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load wav2vec models optimized for high VRAM
        print("Loading wav2vec models...")
        audio_device = self.device if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40 * 1024**3 else 'cpu'
        print(f"Loading audio encoder on: {audio_device}")
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            self.wav2vec_dir, 
            local_files_only=True,
            attn_implementation="eager"
        ).to(audio_device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.wav2vec_dir, 
            local_files_only=True
        )
        self.audio_device = audio_device
        
        # Load MultiTalk pipeline
        print("Loading MultiTalk pipeline...")
        import wan
        from wan.configs import WAN_CONFIGS

        self.cfg = WAN_CONFIGS["multitalk-14B"]
        self.wan_i2v = wan.MultiTalkPipeline(
            config=self.cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False, 
            use_usp=False,
            t5_cpu=True
        )
        
        # GPU optimizations for high-VRAM setup (A100/H100/H200)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🔍 Detected {vram_gb:.1f}GB VRAM")
            
            if vram_gb > 40:  # High VRAM setup
                print("🚀 High-VRAM detected: Enabling maximum performance optimizations")
                # Enable advanced GPU features for maximum speed
                torch.backends.cuda.enable_flash_sdp(True)
                torch.cuda.empty_cache()  # Clear any existing memory
                print(
                    "⚡ Enabled Flash-SDP (cuDNN benchmark / TF32 in predict if high-VRAM)"
                )
            else:
                print("🔧 Standard GPU optimizations enabled")
                torch.backends.cuda.enable_flash_sdp(True)
                torch.cuda.empty_cache()
        
        print("✅ Model setup completed successfully!")

    def predict(
        self,
        image: CogPath = Input(description="Reference image containing the person(s) for video generation"),
        first_audio: CogPath = Input(description="First audio file for driving the conversation"),
        prompt: str = Input(
            description="Text prompt describing the desired interaction or conversation scenario",
            default="A smiling man and woman wearing headphones sit in front of microphones, appearing to host a podcast."
        ),
        second_audio: File = Input(
            description="Second audio file for multi-person conversation (optional)",
            default=None,
        ),
        num_frames: int = Input(
            description="Number of frames to generate (automatically adjusted to nearest valid value of form 4n+1, e.g., 81, 181)",
            default=81,
            ge=25,
            le=201
        ),
        sampling_steps: int = Input(
            description="Number of sampling steps (higher = better quality, lower = faster)",
            default=40,
            ge=2,
            le=100
        ),
        seed: int = Input(
            description="Random seed for reproducible results",
            default=None,
        ),
        turbo: bool = Input(
            description="Enable turbo mode optimizations (adjusts thresholds and guidance scales for speed)",
            default=True
        ),
        person1_bbox: str = Input(
            description="Optional JSON list of 4 floats: [row_min, col_min, row_max, col_max] in pixel indices on the cond image before resize. Matches wan/multitalk.py mask slice human_mask[row_min:row_max, col_min:col_max] with shape [image_height, image_width]. NOT [x1,y1,x2,y2] Cartesian order. If set, person2_bbox is required.",
            default=None,
        ),
        person2_bbox: str = Input(
            description="Same as person1_bbox: [row_min, col_min, row_max, col_max] for person 2. If set, person1_bbox is required.",
            default=None,
        ),
        active_speaker: str = Input(
            description="When using bboxes: which person receives first_audio if inactive_speaker_mode is none; required with person bboxes.",
            default=None,
            choices=["person1", "person2"],
        ),
        inactive_speaker_mode: str = Input(
            description="auto: second stream if second_audio is set, else single-stream slots. none: only active_speaker gets audio (needs bboxes). second_audio: two files (first_audio→person1, second_audio→person2).",
            default="auto",
            choices=["auto", "none", "second_audio"],
        ),
        audio_type: str = Input(
            description="Two-stream mixing for wav2vec prep: para or add. Omit for defaults (para when only one active speaker in two-person mode; add for two-file mode).",
            default=None,
            choices=["para", "add"],
        ),
    ) -> CogPath:
        """Generate a conversational video from audio and reference image"""
        print("VERSION: slot2-lipsync-fix-v1")

        # Optional inputs use non-union types for Cog schema; normalize sentinels for runtime.
        if isinstance(active_speaker, str) and not active_speaker.strip():
            active_speaker = None
        if isinstance(audio_type, str) and not audio_type.strip():
            audio_type = None
        
        # Auto-correct frame count to nearest valid value (4n+1 format)
        original_frames = num_frames
        if (num_frames - 1) % 4 != 0:
            # Find the nearest valid values
            n_lower = (num_frames - 1) // 4
            n_upper = n_lower + 1
            
            frames_lower = 4 * n_lower + 1
            frames_upper = 4 * n_upper + 1
            
            # Choose the closer one
            if abs(num_frames - frames_lower) <= abs(num_frames - frames_upper):
                num_frames = frames_lower
            else:
                num_frames = frames_upper
            
            # Ensure it's within bounds [25, 201]
            num_frames = max(25, min(num_frames, 201))
            
            # Final safety check and adjustment if needed
            while (num_frames - 1) % 4 != 0 and num_frames <= 201:
                num_frames += 1
            
            print(f"📐 Auto-corrected num_frames from {original_frames} to {num_frames} (required format: 4n+1)")
        
        # Validate final bounds
        if num_frames < 25 or num_frames > 201:
            raise ValueError(f"num_frames must be between 25 and 201, got {num_frames}")
        
        # Set random seed
        if seed is None:
            seed = random.randint(0, 99999999)
        
        print(f"🎬 Generating video with seed: {seed}")
        
        p1_bbox = parse_bbox_input(person1_bbox)
        p2_bbox = parse_bbox_input(person2_bbox)
        if (p1_bbox is None) ^ (p2_bbox is None):
            raise ValueError(
                "Both person1_bbox and person2_bbox are required when using bbox mode; omit both for legacy behavior."
            )
        bbox_mode = p1_bbox is not None and p2_bbox is not None
        bbox_img_w = bbox_img_h = 0
        if bbox_mode:
            p1_bbox, p2_bbox, bbox_img_w, bbox_img_h = validate_bboxes_against_cond_image(
                p1_bbox, p2_bbox, str(image)
            )

        inactive_eff = resolve_inactive_speaker_mode(inactive_speaker_mode, second_audio)
        if inactive_eff == "second_audio" and second_audio is None:
            raise ValueError(
                "second_audio is required when inactive_speaker_mode is second_audio."
            )
        if inactive_eff == "none" and second_audio is not None:
            raise ValueError(
                "second_audio was provided but inactive_speaker_mode is none; "
                "use second_audio mode or omit second_audio."
            )

        use_two_person_pipeline = bbox_mode or (inactive_eff == "second_audio")
        if bbox_mode and not active_speaker:
            raise ValueError(
                "active_speaker (person1 or person2) is required when person1_bbox and person2_bbox are set."
            )
        if use_two_person_pipeline and inactive_eff == "none":
            if not bbox_mode:
                raise ValueError(
                    "inactive_speaker_mode=none with two person slots requires person1_bbox and person2_bbox."
                )

        if audio_type is not None and audio_type not in ("para", "add"):
            raise ValueError("audio_type must be para or add when set.")

        if audio_type is None:
            if use_two_person_pipeline and inactive_eff == "none":
                audio_type_eff = "para"
            elif use_two_person_pipeline and inactive_eff == "second_audio":
                audio_type_eff = "add"
            else:
                audio_type_eff = "para"
        else:
            audio_type_eff = audio_type

        logger.info(
            "MultiTalk routing: bbox_mode=%s active_speaker=%s inactive_speaker_mode=%s (effective=%s) audio_type=%s",
            bbox_mode,
            active_speaker,
            inactive_speaker_mode,
            inactive_eff,
            audio_type_eff,
        )
        if bbox_mode:
            logger.info(
                "MultiTalk bbox (final, clamped): PIL cond_image width=%d height=%d; "
                "format [row_min,col_min,row_max,col_max] -> mask[rows,cols] per wan/multitalk.py; "
                "person1_bbox=%s person2_bbox=%s",
                bbox_img_w,
                bbox_img_h,
                p1_bbox,
                p2_bbox,
            )

        speech_slots_info = "single-person: cond_audio person1 only"
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_save_dir = os.path.join(temp_dir, "audio_embeddings")
            os.makedirs(audio_save_dir, exist_ok=True)

            if use_two_person_pipeline:
                use_two_files, assign = resolve_multitalk_audio_assignment(
                    inactive_eff, active_speaker
                )
                if use_two_files:
                    speech_slots_info = (
                        "two-file: person1<-first_audio, person2<-second_audio"
                    )
                else:
                    speech_slots_info = (
                        "person1=real speech, person2=silence embedding"
                        if assign["person1"] == "first"
                        else "person1=silence embedding, person2=real speech"
                    )
                print("🎤 Processing two-person audio slots...")
                if use_two_files:
                    speech1, speech2, combined_speech = audio_prepare_multi(
                        str(first_audio), str(second_audio), audio_type_eff
                    )
                else:
                    speech_in = audio_prepare_single(str(first_audio))
                    silent = np.zeros_like(speech_in)
                    if assign["person1"] == "first":
                        speech1, speech2 = speech_in, silent
                        logger.info(
                            "cond_audio routing (inactive_speaker_mode=none): person1=real speech, person2=silence→embedding"
                        )
                    else:
                        speech1, speech2 = silent, speech_in
                        logger.info(
                            "cond_audio routing (inactive_speaker_mode=none): person1=silence→embedding, person2=real speech"
                        )
                    combined_speech = speech_in

                embedding1 = get_embedding(
                    speech1,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    device=self.audio_device,
                )
                embedding2 = get_embedding(
                    speech2,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    device=self.audio_device,
                )
                emb1_path = os.path.join(audio_save_dir, "1.pt")
                emb2_path = os.path.join(audio_save_dir, "2.pt")
                torch.save(embedding1, emb1_path)
                torch.save(embedding2, emb2_path)

                sum_audio_path = os.path.join(audio_save_dir, "sum.wav")
                sf.write(sum_audio_path, combined_speech, 16000)

                input_data = {
                    "prompt": prompt,
                    "cond_image": str(image),
                    "audio_type": audio_type_eff,
                    "cond_audio": {
                        "person1": emb1_path,
                        "person2": emb2_path,
                    },
                    "video_audio": sum_audio_path,
                }
                if bbox_mode:
                    input_data["bbox"] = build_bbox_payload(p1_bbox, p2_bbox)
            else:
                print("🎤 Processing single-person audio...")
                speech = audio_prepare_single(str(first_audio))
                embedding = get_embedding(
                    speech,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    device=self.audio_device,
                )

                emb_path = os.path.join(audio_save_dir, "1.pt")
                sum_audio_path = os.path.join(audio_save_dir, "sum.wav")

                torch.save(embedding, emb_path)
                sf.write(sum_audio_path, speech, 16000)

                input_data = {
                    "prompt": prompt,
                    "cond_image": str(image),
                    "cond_audio": {
                        "person1": emb_path,
                    },
                    "video_audio": sum_audio_path,
                }

            # Configure generation parameters based on turbo mode and VRAM availability
            high_vram = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40 * 1024**3

            if turbo:
                teacache_thresh = 0.8
                text_guide_scale = 3.0
                audio_guide_scale = 3.0
                shift = 5.0
                offload_model = not high_vram
                print(f"🚀 TURBO MODE: {sampling_steps} steps, thresh={teacache_thresh}")
            else:
                teacache_thresh = 0.3
                text_guide_scale = 5.0
                audio_guide_scale = 4.0
                shift = 7.0
                offload_model = not high_vram  # Don't offload with high VRAM for maximum speed
                print(f"🎬 QUALITY MODE: {sampling_steps} steps{', keeping models in GPU' if high_vram else ''}")

            if high_vram and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Configure optimizations using SimpleNamespace (matching original)
            extra_args = SimpleNamespace(
                use_teacache=True,
                teacache_thresh=teacache_thresh,
                use_apg=False,
                size='multitalk-480'
            )

            bbox_wh_str = (
                f"{bbox_img_w}x{bbox_img_h}"
                if bbox_mode
                else "n/a (bbox_mode=False)"
            )
            print(
                f"[pre-gen] cond_image PIL WxH (bbox validation)={bbox_wh_str} | "
                f"active_speaker={active_speaker!r} | person1_bbox={p1_bbox} | "
                f"person2_bbox={p2_bbox} | audio_slots: {speech_slots_info}"
            )
            print("🎬 Generating video...")

            # Generate video using loaded pipeline (exact parameters from original)
            video = self.wan_i2v.generate(
                input_data,
                size_buckget="multitalk-480",
                motion_frame=25,
                frame_num=num_frames,
                shift=shift,
                sampling_steps=sampling_steps,
                text_guide_scale=text_guide_scale,
                audio_guide_scale=audio_guide_scale,
                seed=seed,
                offload_model=offload_model,
                max_frames_num=num_frames,
                extra_args=extra_args
            )
            
            # Save video (following original save pattern)
            output_name = f"multitalk_{abs(hash(prompt + str(seed))) % 10000}"
            print(f"💾 Saving video...")
            from wan.utils.multitalk_utils import save_video_ffmpeg

            save_video_ffmpeg(video, output_name, [input_data['video_audio']])
            
            # Find and return generated video
            output_file = f"{output_name}.mp4"
            if not os.path.exists(output_file):
                # Look for any mp4 files with our output name
                for file in os.listdir("."):
                    if output_name in file and file.endswith('.mp4'):
                        output_file = file
                        break
                
                if not os.path.exists(output_file):
                    raise RuntimeError(f"Video generation failed - output file not found")
            
            # Copy to permanent location for return
            final_output = f"/tmp/final_{output_name}.mp4"
            shutil.copy2(output_file, final_output)
            
            # Cleanup GPU memory for optimal performance in subsequent runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"✅ Video generation completed: {final_output}")
            return CogPath(final_output)
