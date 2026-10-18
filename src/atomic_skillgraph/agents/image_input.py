"""Explicit image transport limits, separate from text material and token admission."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json

from PIL import Image


def wire_bytes(payload):
    return len(json.dumps(payload, allow_nan=False).encode('utf-8'))


def validate_policy(policy):
    if policy is None:return
    limits={'tokens_per_image','request_bytes','image_bytes','max_images','max_dimension'}
    if (not isinstance(policy,dict) or set(policy)!={'version',*limits}
            or policy['version']!='image-input.v1'
            or any(type(policy[k]) is not int or policy[k]<=0 for k in limits)):
        raise ValueError('Invalid frozen image input policy')


def text_payload(payload):
    """Projection for text-material checks only; never a replacement HTTP payload."""
    value=deepcopy(payload)
    for message in value.get('messages',[]):
        if isinstance(message.get('content'),list):
            message['content']='\n'.join(p['text'] for p in message['content'] if p.get('type')=='text')
    return value


def replace_text(content,text):
    return [{'type':'text','text':text},*[p for p in content if p.get('type')!='text']] if isinstance(content,list) else text


def measure(payload,policy):
    validate_policy(policy)
    full=wire_bytes(payload)
    if policy is None:
        return {'wire_bytes':full,'text_wire_bytes':full,'input_token_bound':full,'images':[]}
    if full>policy['request_bytes']:raise ValueError('Image request exceeds frozen transport limit')
    stripped=deepcopy(payload);images=[]
    for message in stripped.get('messages',[]):
        if not isinstance(message.get('content'),list):continue
        for part in message['content']:
            if part.get('type')!='image_url':continue
            if message.get('role')!='user':raise ValueError('Images require a user message')
            url=part['image_url']['url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise ValueError('Frozen image input requires an inline public image')
            raw=base64.b64decode(url.split(';base64,',1)[1],validate=True)
            if len(raw)>policy['image_bytes']:raise ValueError('Image exceeds frozen byte limit')
            with Image.open(BytesIO(raw)) as image:
                size=image.size;format=image.format
                if format not in {'JPEG','PNG','GIF','WEBP'}:raise ValueError('Unsupported image format')
                if max(size)>policy['max_dimension']:raise ValueError('Image exceeds frozen dimensions')
                image.verify()
            fingerprint=hashlib.sha256(raw).hexdigest()
            images.append({'sha256':fingerprint,'bytes':len(raw),'width':size[0],'height':size[1],
                'format':format,'detail':part['image_url'].get('detail','auto'),
                'token_upper_bound':policy['tokens_per_image']})
            part['image_url']['url']='sha256:'+fingerprint
    if len(images)>policy['max_images']:raise ValueError('Too many images for the frozen policy')
    return {'version':policy['version'],'wire_bytes':full,'text_wire_bytes':wire_bytes(text_payload(payload)),
        'input_token_bound':wire_bytes(stripped)+len(images)*policy['tokens_per_image'],'images':images}
