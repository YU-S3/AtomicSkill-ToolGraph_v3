"""Authorized text selection and bounded grep, with decoded Unicode offsets."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from fnmatch import fnmatchcase
from functools import lru_cache
from copy import deepcopy
import base64
import hashlib
import hmac
import json
import secrets

ALL_PATHS = object()
GREP_SCHEMA = {'type': 'object', 'properties': {'pattern': {'type': 'string'},
    'paths': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['pattern'], 'additionalProperties': False}


def canonical_scope(paths=ALL_PATHS):
    if paths is ALL_PATHS: return None
    if not isinstance(paths, list): raise ValueError('paths must be an array; omit it for the entire corpus')
    normalized = set()
    for name in paths:
        if not isinstance(name, str) or not name.strip(): raise ValueError('paths contains an empty/non-string element')
        path = PurePosixPath(name.replace('\\', '/'))
        if path.is_absolute() or PureWindowsPath(name).drive or '..' in path.parts:
            raise ValueError('Corpus path is not authorized')
        if path.suffix != '.txt': raise ValueError('Corpus path must name a text file')
        normalized.add(path.as_posix())
    return sorted(normalized)


def select_paths(corpus, paths=ALL_PATHS):
    root = Path(corpus).resolve(strict=True)
    if not root.is_dir(): raise RuntimeError('Authorized corpus root is unavailable')
    scope = canonical_scope(paths)
    names = scope if scope is not None else sorted(p.relative_to(root).as_posix() for p in root.rglob('*.txt'))
    selected, parents = {}, {}
    for name in names:
        candidate = root / name
        # Resolve each parent once per selection; individual symlinks still
        # require resolution before the authorization check.
        if candidate.parent not in parents:
            parents[candidate.parent] = candidate.parent.resolve()
        path = parents[candidate.parent] / candidate.name
        if path.is_symlink(): path = path.resolve()
        if not path.is_relative_to(root): raise ValueError('Corpus path is not authorized')
        if not path.is_file(): raise ValueError('Corpus file is missing or not a file')
        if path.suffix != '.txt': raise ValueError('Corpus path must name a text file')
        selected[path.relative_to(root).as_posix()] = path
    return [(name, selected[name]) for name in sorted(selected)]


def _scan(selected, pattern):
    expression = re.compile(pattern, re.I)
    hits = []
    for name, path in selected:
        offset = 0
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(keepends=True)):
            if expression.search(line):
                hits.append({'path': name, 'line': index + 1, 'offset': offset, 'text': line.rstrip('\n')[:1000]})
                if len(hits) == 40:
                    return hits, {'execution_paths': [n for n, _ in selected], 'truncated': True}
            offset += len(line)
    return hits, {'execution_paths': [n for n, _ in selected], 'truncated': False}


@lru_cache(maxsize=128)
def _cached_scan(pattern, identities):
    return _scan([(name, Path(path)) for name, path, _ in identities], pattern)


def grep(corpus, pattern, paths=ALL_PATHS, *, use_cache=True):
    # Re-authorize every query. Cache only bounded query results, not corpus
    # contents; file replacement/modification invalidates the corresponding key.
    selected = select_paths(corpus, paths)
    if not use_cache:
        return _scan(selected, pattern)
    identities = []
    for name, path in selected:
        stat = path.stat()
        identities.append((name, str(path), (stat.st_dev, stat.st_ino, stat.st_size,
                                           stat.st_mtime_ns, stat.st_ctime_ns)))
    return deepcopy(_cached_scan(pattern, tuple(identities)))


def glob_paths(corpus, pattern):
    """POSIX segment matching over the same authorized set used by grep."""
    if not isinstance(pattern, str) or not pattern.strip(): raise ValueError('Empty glob pattern')
    pattern = pattern.replace('\\', '/')
    path = PurePosixPath(pattern)
    if path.is_absolute() or PureWindowsPath(pattern).drive or '..' in path.parts:
        raise ValueError('Corpus pattern is not authorized')
    parts = path.parts
    def matches(name):
        segments = PurePosixPath(name).parts
        @lru_cache(None)
        def match(i, j):
            if i == len(parts): return j == len(segments)
            if parts[i] == '**': return match(i + 1, j) or (j < len(segments) and match(i, j + 1))
            return j < len(segments) and fnmatchcase(segments[j], parts[i]) and match(i + 1, j + 1)
        return match(0, 0)
    return [name for name, _ in select_paths(corpus) if matches(name)]


RETRIEVAL_VERSION = 'office.retrieval.v3'
GREP_PAGE_SCHEMA = {'type':'object','properties':{**GREP_SCHEMA['properties'],
    'cursor':{'type':['string','null']}, 'limit':{'type':'integer','minimum':1,'maximum':40}},
    'required':['pattern'],'additionalProperties':False}
CONTEXT_SCHEMA = {'type':'object','properties':{'path':{'type':'string'},
    'offset':{'type':'integer','minimum':0}, 'before':{'type':'integer','minimum':0,'maximum':12000},
    'after':{'type':'integer','minimum':0,'maximum':12000}},
    'required':['path','offset','before','after'],'additionalProperties':False}


class RetrievalSession:
    """Task-local authenticated pages. Every access rechecks authorization and file identity."""
    def __init__(self, corpus, *, key=None):
        self.corpus = Path(corpus).resolve(strict=True)
        self.key = key or secrets.token_bytes(32)
        self.cache = {}
        self.directory = []
        self.directories = {}

    @staticmethod
    def identity(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def catalog(self):
        if not self.directories or any(not p.exists() or self.identity(p) != v for p,v in self.directories.items()):
            self.directory = [n for n,_ in select_paths(self.corpus)]
            self.directories = {p:self.identity(p) for p in [self.corpus,*self.corpus.rglob('*')] if p.is_dir()}
        # Resolving an exact list also detects changed symlinks without relying on directory mtimes.
        selected = select_paths(self.corpus, self.directory)
        identities = [(n,self.identity(p)) for n,p in selected]
        identity = hashlib.sha256(json.dumps([str(self.corpus),identities],separators=(',',':')).encode()).hexdigest()
        return selected, identity

    def page(self, pattern, paths=ALL_PATHS, cursor=None, limit=40):
        if not isinstance(pattern,str) or type(limit) is not int or not 1<=limit<=40:
            raise ValueError('Invalid bounded search parameters')
        expression = re.compile(pattern,re.I)
        _, corpus_identity = self.catalog()
        selected = select_paths(self.corpus,paths)
        query = hashlib.sha256(json.dumps([RETRIEVAL_VERSION,corpus_identity,pattern,
            canonical_scope(paths),limit],separators=(',',':')).encode()).hexdigest()
        position = [0,0]
        if cursor is not None:
            try:
                body, signature = cursor.split('.')
                raw = base64.urlsafe_b64decode(body + '='*(-len(body)%4))
                if not hmac.compare_digest(hmac.new(self.key,raw,hashlib.sha256).hexdigest(),signature):
                    raise ValueError('Cursor authentication failed')
                value = json.loads(raw)
                if value['query'] != query: raise ValueError('Cursor query/corpus/scope changed')
                position = value['position']
                if len(position)!=2 or any(type(v) is not int or v<0 for v in position):raise ValueError('Invalid cursor position')
            except (ValueError,KeyError,TypeError,AttributeError) as exc:
                raise ValueError('Invalid or expired search cursor') from exc
        cache_key=(query,tuple(position))
        if cache_key in self.cache:
            result,audit=deepcopy(self.cache[cache_key]);audit['cache_hit']=True
            return result,audit
        hits=[];scanned=[];following=None
        for file_index,(name,path) in enumerate(selected):
            if file_index<position[0]:continue
            scanned.append(name);offset=0
            for line_index,line in enumerate(path.read_text(encoding='utf-8').splitlines(keepends=True)):
                if file_index==position[0] and line_index<position[1]:offset+=len(line);continue
                if expression.search(line):
                    hits.append({'path':name,'line':line_index+1,'offset':offset,'text':line.rstrip('\n')[:1000]})
                    if len(hits)==limit:
                        following=[file_index,line_index+1];break
                offset+=len(line)
            if following is not None:break
        next_cursor=None
        if self.catalog()[1]!=corpus_identity:raise ValueError('Corpus changed during search; restart query')
        if following is not None:
            raw=json.dumps({'query':query,'position':following},separators=(',',':')).encode()
            next_cursor=base64.urlsafe_b64encode(raw).decode().rstrip('=')+'.'+hmac.new(self.key,raw,hashlib.sha256).hexdigest()
        result={'hits':hits,'next_cursor':next_cursor,'complete':next_cursor is None}
        audit={'version':RETRIEVAL_VERSION,'corpus_identity':corpus_identity,'raw_paths':None if paths is ALL_PATHS else paths,
            'scope_omitted':paths is ALL_PATHS,'execution_paths':[n for n,_ in selected], 'scanned_paths':scanned,
            'continued':cursor is not None,'truncated':next_cursor is not None,'cache_hit':False}
        if len(self.cache)>=128:self.cache.clear()
        self.cache[cache_key]=deepcopy((result,audit))
        return result,audit

    def context(self,path,offset,before,after):
        if any(type(v) is not int or v<0 for v in (offset,before,after)) or not 0<before+after<=12000:
            raise ValueError('Context window must contain 1..12000 Unicode characters')
        selected=select_paths(self.corpus,[path]);name,source=selected[0]
        text=source.read_text(encoding='utf-8')
        if offset>len(text):raise ValueError('Context anchor is outside the file')
        start=max(0,offset-before);end=min(len(text),offset+after)
        return {'path':name,'text':text[start:end],'start_offset':start,'end_offset':end,'anchor_offset':offset,
                'next_offset':end if end<len(text) else None}
