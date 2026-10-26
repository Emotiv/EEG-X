"""Keep subjects linked by identical EEG in one partition without dropping rows."""

import hashlib

import numpy as np


def subject_groups(records, index):
    used = set(index["record"].tolist())
    parents = {records[i]["subject"]: records[i]["subject"] for i in used}

    def root(subject):
        while parents[subject] != subject:
            parents[subject] = parents[parents[subject]]
            subject = parents[subject]
        return subject

    def link(left, right):
        left, right = sorted((root(left), root(right)))
        parents[right] = left

    recordings, windows = {}, {}
    for number in sorted(used):
        record = records[number]
        subject = record["subject"]
        previous = recordings.setdefault(record["sha256"], subject)
        link(subject, previous)
    for row in index:
        subject = records[int(row["record"])]["subject"]
        previous = windows.setdefault(bytes(row["signal_sha256"]), subject)
        link(subject, previous)
    return {subject: root(subject) for subject in sorted(parents)}


def assignments(groups, seed, *, source_only=False):
    members = {}
    for subject, group in groups.items():
        members.setdefault(group, []).append(subject)
    minimum = 2 if source_only else 3
    if len(members) < minimum:
        raise ValueError(f"Independent partitions require at least {minimum} duplicate groups")
    ordered = sorted(members, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest())
    total = len(groups)

    def take(pool, target, reserve):
        cumulative = np.cumsum([len(members[group]) for group in pool])
        count = min(
            range(1, len(pool) - reserve + 1),
            key=lambda size: (abs(int(cumulative[size - 1]) - target), size),
        )
        return pool[:count], pool[count:]

    test, pool = ([], ordered) if source_only else take(ordered, max(1, round(total * 0.2)), 2)
    validation, train = take(pool, max(1, round(total * (0.2 if source_only else 0.1))), 1)
    return {
        subject: role
        for role, selected in (("train", train), ("validation", validation), ("test", test))
        for group in selected
        for subject in members[group]
    }


def validate_membership(groups, subjects):
    if set(groups) != set(subjects):
        raise ValueError("Saved subject groups do not cover the selected subjects")
    roles = {}
    for subject, group in groups.items():
        role = subjects[subject]
        if roles.setdefault(group, role) != role:
            raise ValueError("Duplicate-linked subjects cross partitions")
