import io
import math
from dataclasses import dataclass
import logging
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


MIN_WIDTH = 8
MIN_HEIGHT = 8

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    convo_image: bytes | None
    dark_mode_image: bytes | None
    light_mode_image: bytes | None


def _resize_and_pad(
    image: Image.Image, tile_size: int, high_quality_jpeg: bool = False
) -> bytes:
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = tile_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = tile_size
        new_width = int(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return _padded_image(resized_image, high_quality_jpeg)


def resize_tile(image_bytes: bytes, tile_size: int) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as image:
            return _resize_and_pad(image, tile_size)


def _has_transparency(img: Image.Image) -> bool:
    if "A" in img.mode:
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def _apply_clahe_pil(
    image: Image.Image,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    arr = np.array(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def _boost_saturation(image: Image.Image, factor: float = 1.8) -> Image.Image:
    return ImageEnhance.Color(image).enhance(factor)


def _sharpen(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def _adaptive_brightness_correction(
    image: Image.Image, strength: float = 0.25
) -> Image.Image:
    arr = np.array(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    x = np.arange(256, dtype=np.float32) / 255.0
    adjusted = x + strength * (0.5 - x)
    lut = np.clip(adjusted * 255, 0, 255).astype(np.uint8)

    l_adjusted = cv2.LUT(l_channel, lut)
    merged = cv2.merge((l_adjusted, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def enhance_image_with_clahe(image_bytes: bytes, tile_size: int | None) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as rgb_img:
            enhanced = _apply_clahe_pil(rgb_img)
            enhanced = _adaptive_brightness_correction(enhanced)
            enhanced = _boost_saturation(enhanced, factor=1.5)
            enhanced = _sharpen(enhanced)
            if tile_size is None:
                return _padded_image(enhanced, high_quality_jpeg=True)
            return _resize_and_pad(enhanced, tile_size, high_quality_jpeg=True)


def process_image_bytes(image_bytes: bytes, tile_size: int) -> ProcessedImage:
    with Image.open(io.BytesIO(image_bytes)) as img:
        if _has_transparency(img):
            logger.info(f"Image has transparency (mode={img.mode})")
            img_rgba = img.convert("RGBA")
            black_bg = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
            black_comp = Image.alpha_composite(black_bg, img_rgba)
            with black_comp.convert("RGB") as black_rgb_img:
                dark_mode = _resize_and_pad(black_rgb_img, tile_size)
            white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            white_comp = Image.alpha_composite(white_bg, img_rgba)
            with white_comp.convert("RGB") as white_rgb_img:
                light_mode = _resize_and_pad(white_rgb_img, tile_size)
        else:
            logger.info("Image does not have transparency")
            dark_mode = None
            light_mode = None
        with img.convert("RGB") as rgb_img:
            resized = _resize_and_pad(rgb_img, tile_size)
            return ProcessedImage(
                convo_image=resized,
                dark_mode_image=dark_mode,
                light_mode_image=light_mode,
            )


def _img_to_bytes(image: Image.Image, high_quality_jpeg: bool = False) -> bytes:
    with io.BytesIO() as output:
        if high_quality_jpeg:
            image.save(output, format="JPEG", quality=95)
        else:
            image.save(output, format="JPEG")
        return output.getvalue()


def _padded_image(image: Image.Image, high_quality_jpeg: bool = False) -> bytes:
    width, height = image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        new_width = max(width, MIN_WIDTH)
        new_height = max(height, MIN_HEIGHT)
        with Image.new("RGB", (new_width, new_height), (0, 0, 0)) as padded_image:
            padded_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return _img_to_bytes(padded_image, high_quality_jpeg)
    return _img_to_bytes(image, high_quality_jpeg)


def pad_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as image:
            return _padded_image(image)


_MOTION_REVEAL_MIN_FRAMES = 4
_MOTION_REVEAL_MAX_INPUT_FRAMES = 32
_MOTION_REVEAL_WORK_MAX_DIM = 640
_MOTION_REVEAL_COVER_WEIGHT = 0.7
_MOTION_REVEAL_GRID_FRAMES = 4
_MOTION_REVEAL_GRID_COLUMNS = 2
_MOTION_REVEAL_JPEG_QUALITY = 90
_MOTION_REVEAL_MIN_RESIDUAL_P99 = 3.0
_MOTION_REVEAL_MAX_RESIDUAL_P99 = 75.0


def _decode_reveal_frame(frame_bytes: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = _MOTION_REVEAL_WORK_MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img.astype(np.float32)


def _stretch_to_u8(arr: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi <= lo + 1e-3:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _encode_reveal(bgr: np.ndarray) -> bytes | None:
    ok, jpeg = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _MOTION_REVEAL_JPEG_QUALITY]
    )
    return jpeg.tobytes() if ok else None


def _skin_fraction(bgr_u8: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb)
    mask = (
        (ycrcb[:, :, 0] > 60)
        & (ycrcb[:, :, 1] > 135)
        & (ycrcb[:, :, 1] < 175)
        & (ycrcb[:, :, 2] > 80)
        & (ycrcb[:, :, 2] < 130)
    )
    return float(mask.mean())


def build_motion_reveal_images(frame_jpegs: list[bytes]) -> list[bytes]:
    if len(frame_jpegs) < _MOTION_REVEAL_MIN_FRAMES:
        return []
    if len(frame_jpegs) > _MOTION_REVEAL_MAX_INPUT_FRAMES:
        indices = np.linspace(
            0, len(frame_jpegs) - 1, _MOTION_REVEAL_MAX_INPUT_FRAMES
        ).astype(int)
        frame_jpegs = [frame_jpegs[i] for i in indices]

    decoded: list[np.ndarray] = []
    target_hw: tuple[int, int] | None = None
    for frame_bytes in frame_jpegs:
        img = _decode_reveal_frame(frame_bytes)
        if img is None:
            continue
        if target_hw is None:
            target_hw = (img.shape[0], img.shape[1])
        elif (img.shape[0], img.shape[1]) != target_hw:
            img = cv2.resize(
                img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA
            )
        decoded.append(img)
    if len(decoded) < _MOTION_REVEAL_MIN_FRAMES:
        return []

    stack = np.stack(decoded, axis=0)
    median = np.median(stack, axis=0)
    residual = np.abs(stack - median[None]).mean(axis=-1)
    per_frame_p99 = np.percentile(residual.reshape(len(decoded), -1), 99, axis=1)
    if (
        not _MOTION_REVEAL_MIN_RESIDUAL_P99
        <= float(np.median(per_frame_p99))
        <= _MOTION_REVEAL_MAX_RESIDUAL_P99
    ):
        return []
    motion_energy = residual.mean(axis=(1, 2))

    unmixed = [
        _stretch_to_u8(frame - _MOTION_REVEAL_COVER_WEIGHT * median) for frame in stack
    ]
    skin = np.array([_skin_fraction(still) for still in unmixed])
    order = np.argsort(-(motion_energy * (0.5 + skin)))

    out: list[bytes] = []

    best = int(order[0])
    pair = np.concatenate([stack[best].astype(np.uint8), unmixed[best]], axis=1)
    encoded = _encode_reveal(pair)
    if encoded:
        out.append(encoded)

    encoded = _encode_reveal(unmixed[best])
    if encoded:
        out.append(encoded)

    grid_indices = sorted(int(i) for i in order[:_MOTION_REVEAL_GRID_FRAMES])
    tiles = [unmixed[i] for i in grid_indices]
    h, w = tiles[0].shape[:2]
    cols = _MOTION_REVEAL_GRID_COLUMNS
    rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = tile
    encoded = _encode_reveal(canvas)
    if encoded:
        out.append(encoded)

    return out
