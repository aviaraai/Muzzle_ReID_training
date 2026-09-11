"""P x K batch sampler with appearance-cluster hard negatives.

Batch 32 = P=8 identities x K=4 images. Six of the eight identities are drawn
from ONE appearance cluster, the other two at random from anywhere.

Why: the goal is an encoder that reads the muzzle print rather than the
animal's appearance. If a batch holds eight animals of eight different colours
the loss is separable on colour alone and the model never has to look at the
print. Putting 24 of 32 images in same-coat competition removes that shortcut,
while the two random identities keep some easy negatives so early training
still has usable gradient.

Clusters come from k-means over stock ResNet50 identity centroids -- the
demonstrated appearance channel (0.8619 top-1 on the benchmark, and flat under
detail destruction all the way to R=56, i.e. it reads coarse appearance and
ignores the ridge band). No breed or colour labels exist in the 300-corpus, so
this is the only available grouping, and it was visually validated: one
cluster is 57 uniformly black animals, another 44 white/pale ones.

Cluster choice is weighted by size so that the largest clusters -- where
lookalike confusion actually concentrates -- are sampled most often.
Singleton clusters cannot supply six identities and fall back to a random
draw.
"""
from __future__ import annotations

import random
from typing import Iterator


class PKClusterSampler:
    """Yields lists of dataset indices, one list per batch."""

    def __init__(
        self,
        indices_by_identity: dict[str, list[int]],
        cluster_of_identity: dict[str, int],
        P: int = 8,
        K: int = 4,
        same_cluster: int = 6,
        seed: int = 20260911,
        batches: int | None = None,
    ) -> None:
        self.idx = {k: list(v) for k, v in indices_by_identity.items() if v}
        self.ids = sorted(self.idx)
        self.cluster = cluster_of_identity
        self.P, self.K, self.same = P, K, same_cluster
        self.rng = random.Random(seed)

        self.by_cluster: dict[int, list[str]] = {}
        for i in self.ids:
            self.by_cluster.setdefault(self.cluster.get(i, -1), []).append(i)
        for v in self.by_cluster.values():
            v.sort()
        # only clusters that can actually supply `same` identities are eligible
        self.eligible = [c for c, v in self.by_cluster.items() if len(v) >= same_cluster]
        self.weights = [len(self.by_cluster[c]) for c in self.eligible]

        n_images = sum(len(v) for v in self.idx.values())
        self.batches = batches if batches is not None else max(1, n_images // (P * K))

    def __len__(self) -> int:
        return self.batches

    def _pick_identities(self) -> list[str]:
        chosen: list[str] = []
        if self.eligible:
            c = self.rng.choices(self.eligible, weights=self.weights, k=1)[0]
            pool = self.by_cluster[c]
            chosen = self.rng.sample(pool, min(self.same, len(pool)))
        rest = [i for i in self.ids if i not in set(chosen)]
        need = self.P - len(chosen)
        if need > 0:
            chosen += self.rng.sample(rest, min(need, len(rest)))
        return chosen[: self.P]

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.batches):
            batch: list[int] = []
            for ident in self._pick_identities():
                pool = self.idx[ident]
                # sample K without replacement where possible; identities with
                # fewer than K images repeat rather than shrink the batch
                if len(pool) >= self.K:
                    batch += self.rng.sample(pool, self.K)
                else:
                    batch += [self.rng.choice(pool) for _ in range(self.K)]
            yield batch
