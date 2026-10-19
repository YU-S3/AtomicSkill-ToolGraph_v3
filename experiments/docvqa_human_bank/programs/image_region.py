"""Image ROI tool for the actual DocVQA public image, not an OCR system.

Input path must be an AnswerAdapter worker_resources path mounted read-only at
/workspace/inputs/image_N.ext. It returns a bounded JPEG for the runtime's
multimodal bridge, plus dimensions and the exact region used.
"""
import base64
import hashlib
from io import BytesIO
from pathlib import Path
import re

from PIL import Image, ImageEnhance, ImageOps


def crop_bytes(raw, bbox, max_edge=1280, enhancement='none'):
    """Pure transform used by the Worker and by zero-HTTP fixture tests."""
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(type(x) is not int or not 0 <= x <= 1000 for x in bbox) or
            not (bbox[0] < bbox[2] and bbox[1] < bbox[3])):
        raise ValueError('bbox must be [left,top,right,bottom] in [0,1000], nonempty')
    if type(max_edge) is not int or not 384 <= max_edge <= 1600:
        raise ValueError('max_edge must be 384..1600')
    if enhancement not in ('none', 'contrast', 'grayscale'):
        raise ValueError('Unsupported enhancement')
    with Image.open(BytesIO(raw)) as original:
        original.load()
        if original.width * original.height > 75_000_000:
            raise ValueError('Image is too large for a bounded crop')
        frame = ImageOps.exif_transpose(original).convert('RGB')
    width, height = frame.size
    x0 = max(0, min(width-1, round(width*bbox[0]/1000)))
    x1 = max(x0+1, min(width, round(width*bbox[2]/1000)))
    y0 = max(0, min(height-1, round(height*bbox[1]/1000)))
    y1 = max(y0+1, min(height, round(height*bbox[3]/1000)))
    region = frame.crop((x0,y0,x1,y1))
    if enhancement == 'grayscale':
        region = ImageOps.grayscale(region).convert('RGB')
    elif enhancement == 'contrast':
        region = ImageEnhance.Contrast(region).enhance(1.25)
    region.thumbnail((max_edge,max_edge), Image.Resampling.LANCZOS)
    for quality in (82,70,58):
        stream = BytesIO()
        region.save(stream,format='JPEG',quality=quality,optimize=True)
        image_bytes = stream.getvalue()
        encoded = base64.b64encode(image_bytes).decode('ascii')
        if len(encoded) <= 680_000:
            return {'crop_base64':encoded,'image_mime':'image/jpeg',
                    'image_sha256':hashlib.sha256(image_bytes).hexdigest(),
                    'source_sha256':hashlib.sha256(raw).hexdigest(),
                    'source_width':width,'source_height':height,
                    'crop_width':region.width,'crop_height':region.height,
                    'source_bbox_pixels':[x0,y0,x1,y1],
                    'bbox_normalized':bbox,
                    'scope':'crop_of_current_authorized_public_image_not_ocr'}
    raise ValueError('Compressed crop still exceeds the RPC payload bound')


def run(ctx, inputs):
    try:
        path = inputs['image_path']
        if (not isinstance(path,str) or not re.fullmatch(
                r'/workspace/inputs/image_\d+\.(?:png|jpg|jpeg|webp|gif|bmp)',path,re.I)):
            raise ValueError('Only an advertised /workspace/inputs/image_N.ext is allowed')
        source = Path(path)
        if not source.is_file() or source.is_symlink():
            raise ValueError('Authorized public image is not mounted')
        raw = source.read_bytes()
        if len(raw) > 33_554_432:
            raise ValueError('Input exceeds DocVQA frozen image-byte policy')
        outputs = crop_bytes(raw, inputs['bbox'],inputs.get('max_edge',1280),
                             inputs.get('enhancement','none'))
        return {'status':'ok','outputs':outputs}
    except (KeyError,ValueError,OSError,TypeError) as exc:
        return {'status':'needs_input','outputs':{},'detail':str(exc)[:400]}
