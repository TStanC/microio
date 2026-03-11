"""Access helpers for microio extension metadata blocks."""

from __future__ import annotations

import logging

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.extras")


def read_microio_extras(ds: DatasetHandle, scene_id: str) -> dict:
    """Read the ``microio`` extension block from one scene.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene identifier under the root group.

    Returns
    -------
    dict
        The stored ``microio`` metadata block, or an empty dictionary if absent.
    """
    attrs = ds.root[scene_id].attrs.asdict()
    logger.debug("Reading microio extras for scene %s", scene_id)
    return attrs.get("microio", {})
