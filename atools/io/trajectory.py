import io
import os

import ase.io
import lmdb
import numpy as np
from ase import Atoms


class LMDBTrajectory:
    def __init__(self, filename: str):
        self.filename = filename
        self.db = lmdb.open(
            filename,
            subdir=False,
            readonly=True,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        self._length = None

    def __len__(self):
        if self._length is None:
            with self.db.begin() as txn:
                self._length = txn.stat()["entries"]
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._get_single_item(idx)
        elif isinstance(idx, slice):
            return self._get_slice(idx)
        elif isinstance(idx, (list, np.ndarray)):
            return self._get_multiple_items(idx)
        else:
            raise TypeError("Invalid index type. Use int, slice, list, or numpy array.")

    def _get_single_item(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        with self.db.begin() as txn:
            key = str(idx).encode()
            bio = io.BytesIO(txn.get(key))
            bio.seek(0)
            return ase.io.read(bio, format="traj")

    def _get_slice(self, idx):
        indices = range(*idx.indices(len(self)))
        return [self._get_single_item(i) for i in indices]

    def _get_multiple_items(self, indices):
        return [self._get_single_item(i) for i in indices]

    @classmethod
    def write(cls, filename, images, mode="a"):
        if mode not in ("w", "a"):
            raise ValueError("mode must be one of 'w', 'a'")

        if mode == "w" and os.path.exists(filename):
            os.remove(filename)

        if isinstance(images, Atoms):
            images = [images]

        db = lmdb.open(
            filename,
            subdir=False,
            readonly=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )

        idx = db.stat()["entries"]
        with db.begin(write=True) as txn:
            for offset, atoms in enumerate(images):
                key = str(idx + offset).encode()
                bio = io.BytesIO()
                ase.io.write(bio, atoms, format="traj")
                bio.seek(0)
                txn.put(key, bio.read())

        db.close()