"""Acquire hash-locked official runtime files; no installation or code changes."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent
COMMIT=(ROOT/'OFFICIAL_COMMIT').read_text().strip()
UPSTREAM=ROOT/'vendor/mqe'


def blob_sha(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def sources():
    for line in (ROOT/'SOURCE_BLOBS').read_text().splitlines():
        if not line or line.startswith('#'):continue
        sha,name=line.split(' ',1)
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        yield sha,name


def verify():
    for sha,name in sources():
        assert blob_sha((UPSTREAM/name).read_bytes())==sha,name


def acquire():
    # Reuse only byte-identical blobs from the already-pinned official benchmark.
    original=ROOT.parent/'ogbench_official_qrl/vendor/ogbench'
    counts=dict(existing=0,local_exact_blob=0,downloaded=0)
    for sha,name in sources():
        target=UPSTREAM/name
        if target.exists():
            assert blob_sha(target.read_bytes())==sha,name
            counts['existing']+=1;continue
        local=original/name
        if local.is_file() and blob_sha(local.read_bytes())==sha:
            data=local.read_bytes();counts['local_exact_blob']+=1
        else:
            url=f'https://raw.githubusercontent.com/WJ2003B/mqe-release/{COMMIT}/{name}'
            with urlopen(url,timeout=60) as response:data=response.read()
            counts['downloaded']+=1
        assert blob_sha(data)==sha,name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
    verify();print(counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify();print('All 90 locked official runtime files match Git blob hashes.')
    else:acquire()
