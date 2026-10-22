import json

import numpy as np
import pytest

from pretrain.channel_identity import ChannelVocabularyError, load_channel_vocabulary


def test_electrode_aliases_share_one_physical_identity():
    vocabulary = load_channel_vocabulary()
    assert vocabulary.resolve_names(["T3", "T7"]).tolist()[0] == vocabulary.resolve_names(["T7"])[0]
    names = vocabulary.canonical_names(["T3", "T7"])
    assert names[0] == names[1]
    assert vocabulary.resolve_names(["  pZ "]).tolist() == vocabulary.resolve_names(["Pz"]).tolist()
    with pytest.raises(ChannelVocabularyError, match="Channels missing"):
        vocabulary.resolve_names(["not_an_electrode"])


def test_channel_positions_resolve_to_the_same_stable_identities():
    vocabulary = load_channel_vocabulary()
    records = vocabulary.channels[:3]
    positions = np.asarray([record.position for record in records])
    assert vocabulary.resolve_positions(positions).tolist() == [record.index for record in records]


def test_catalog_rejects_an_alias_that_changes_physical_position(tmp_path):
    vocabulary = load_channel_vocabulary()
    payload = json.loads(vocabulary.source_path.read_text())
    record = next(record for record in payload["channels"] if record["aliases"])
    record["aliases"][0]["position"][0] += 1
    path = tmp_path / "invalid-vocabulary.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ChannelVocabularyError, match="position must equal canonical"):
        load_channel_vocabulary(path)
