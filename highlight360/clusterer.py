"""身份聚类: 把多条轨迹的 embedding 合并成人物身份。

优先用 scipy 层次聚类 (余弦距离 + 阈值切割); 未安装时降级为
基于阈值的贪心并查集聚类。二者都不需要预先知道人数。
"""
from __future__ import annotations

import numpy as np

from .config import IdentityConfig
from .types import Person, Track


def _cosine_distance_matrix(embs: np.ndarray) -> np.ndarray:
    sim = embs @ embs.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def _greedy_cluster(embs: np.ndarray, threshold: float) -> list[int]:
    """并查集式贪心聚类: 距离 < threshold 即合并。"""
    n = len(embs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dist = _cosine_distance_matrix(embs)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < threshold:
                parent[find(i)] = find(j)
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels


class IdentityClusterer:
    def __init__(self, cfg: IdentityConfig):
        self.cfg = cfg

    def cluster(self, tracks: list[Track]) -> list[Person]:
        if not tracks:
            return []
        embs = np.stack([
            t.embedding if t.embedding is not None else np.zeros(48, np.float32)
            for t in tracks
        ])
        labels = self._run(embs)
        # 组装 Person
        people: dict[int, Person] = {}
        for track, label in zip(tracks, labels):
            track.person_id = label
            p = people.setdefault(label, Person(person_id=label))
            p.track_ids.append(track.track_id)
        # 按 person_id 重排为连续 0..k
        ordered = sorted(people.values(), key=lambda p: p.person_id)
        remap = {p.person_id: i for i, p in enumerate(ordered)}
        for track in tracks:
            track.person_id = remap[track.person_id]
        for p in ordered:
            p.person_id = remap[p.person_id]
        return ordered

    def _run(self, embs: np.ndarray) -> list[int]:
        if len(embs) <= 1:
            return [0] * len(embs)
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
            dist = _cosine_distance_matrix(embs)
            condensed = squareform(dist, checks=False)
            z = linkage(condensed, method="average")
            if self.cfg.n_people > 0:
                labels = fcluster(z, t=self.cfg.n_people, criterion="maxclust")
            else:
                labels = fcluster(z, t=self.cfg.distance_threshold,
                                 criterion="distance")
            return [int(x) for x in labels]
        except Exception as e:
            print(f"[cluster] scipy 不可用, 用贪心聚类降级: {e}")
            return _greedy_cluster(embs, self.cfg.distance_threshold)
