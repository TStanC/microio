from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_OUT = REPO_ROOT.parent / "data_out"
TEST_FIXTURES_ROOT = DATA_OUT / "test_fixtures"

AXES = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def fresh_dataset_path(prefix: str, *, root: Path | None = None) -> Path:
    base = root or TEST_FIXTURES_ROOT
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path


def cleanup_dataset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def create_lif_like_dataset(
    path: Path,
    *,
    scene_ids: list[str],
    include_invalid_scene_zero: bool = False,
) -> Path:
    root = zarr.open(path, mode="w", zarr_format=2)
    for scene_id in scene_ids:
        if scene_id == "0" and include_invalid_scene_zero:
            _create_lif_invalid_scene(root, scene_id)
            continue
        _create_lif_valid_scene(root, scene_id)

    ome = root.create_group("OME")
    ome.attrs["series"] = [str(scene_id) for scene_id in scene_ids]
    (path / "OME" / "METADATA.ome.xml").write_text(_lif_ome_xml(), encoding="utf-8")
    return path


def create_vsi_like_dataset(path: Path) -> Path:
    root = zarr.open(path, mode="w", zarr_format=2)
    scene = root.create_group("0")
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": "C555",
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": path_name,
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.25 * (2**level), 0.25 * (2**level)]}
                            ],
                        }
                        for level, path_name in enumerate(["0", "1", "2", "3", "4"])
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {
                        "active": True,
                        "coefficient": 1,
                        "color": "00FF00",
                        "family": "linear",
                        "inverted": False,
                        "label": "DNA",
                        "window": {"min": 0.0, "max": 22800.0, "start": 10.0, "end": 2048.0},
                    }
                ],
                "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "greyscale"},
            },
        }
    )
    scene.create_array("0", shape=(100, 1, 34, 32, 32), dtype=np.uint16, chunks=(1, 1, 1, 16, 16), write_data=False)
    scene.create_array("1", shape=(100, 1, 34, 16, 16), dtype=np.uint16, chunks=(1, 1, 1, 8, 8), write_data=False)
    scene.create_array("2", shape=(100, 1, 34, 8, 8), dtype=np.uint16, chunks=(1, 1, 1, 4, 4), write_data=False)
    scene.create_array("3", shape=(100, 1, 34, 4, 4), dtype=np.uint16, chunks=(1, 1, 1, 2, 2), write_data=False)
    scene.create_array("4", shape=(100, 1, 34, 2, 2), dtype=np.uint16, chunks=(1, 1, 1, 1, 1), write_data=False)
    scene["0"][0, 0, 0, 0:2, 0:2] = np.array([[11, 99], [255, 1024]], dtype=np.uint16)

    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(_vsi_ome_xml(), encoding="utf-8")
    return path


def create_label_timepoint_dataset(path: Path) -> Path:
    root = zarr.open(path, mode="w", zarr_format=2)
    scene = root.create_group("0")
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": "timecourse",
                    "axes": AXES,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.5, 0.5, 0.5]}],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.5, 1.0, 1.0]}],
                        },
                    ],
                }
            ]
        }
    )
    scene.create_array("0", shape=(8, 1, 8, 64, 64), dtype=np.uint16, chunks=(1, 1, 1, 32, 32), write_data=False)
    scene.create_array("1", shape=(8, 1, 8, 32, 32), dtype=np.uint16, chunks=(1, 1, 1, 16, 16), write_data=False)
    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image Name="timecourse"><Pixels SizeT="8" SizeC="1" SizeZ="8" SizeY="64" SizeX="64"/></Image>'
            "</OME>"
        ),
        encoding="utf-8",
    )
    return path


def create_large_metadata_fixture(path: Path, *, image_count: int = 250, plane_count: int = 32) -> Path:
    root = zarr.open(path, mode="w", zarr_format=2)
    scene = root.create_group("0")
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": "scene_0",
                    "axes": AXES,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                        }
                    ],
                }
            ]
        }
    )
    scene.create_array("0", shape=(1, 1, 1, 4, 4), dtype=np.uint16, chunks=(1, 1, 1, 4, 4), write_data=False)
    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    images: list[str] = []
    for image_index in range(image_count):
        planes = "".join(
            f'<Plane TheT="0" TheC="0" TheZ="{plane_index}" PositionZ="{plane_index}" PositionZUnit="Âµm"/>'
            for plane_index in range(plane_count)
        )
        images.append(
            f'<Image Name="scene_{image_index}"><Pixels SizeT="1" SizeC="1" SizeZ="{plane_count}" SizeY="4" SizeX="4">{planes}</Pixels></Image>'
        )
    xml = '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">' + "".join(images) + "</OME>"
    (path / "OME" / "METADATA.ome.xml").write_text(xml, encoding="utf-8")
    return path


def _create_lif_invalid_scene(root, scene_id: str) -> None:
    scene = root.create_group(scene_id)
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": "Graph Scene",
                    "axes": AXES,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 2.0, 2.0]}],
                        },
                    ],
                }
            ]
        }
    )
    scene.create_array("0", shape=(1, 1, 1, 4, 4), dtype=np.uint8, chunks=(1, 1, 1, 4, 4), write_data=False)


def _create_lif_valid_scene(root, scene_id: str) -> None:
    scene = root.create_group(scene_id)
    name_map = {
        "14": "E8Flex/Stripes/300_300",
        "15": "E8Flex/Stripes/300_300_Merged",
        "16": "E8Flex/Stripes/300_300",
    }
    name = name_map.get(str(scene_id), f"Scene {scene_id}")
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": name,
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5, 0.5]}],
                        }
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {"label": "c0", "color": "00FF00", "window": {"min": 0.0, "max": 125.0, "start": 1.0, "end": 100.0}},
                    {"label": "c1", "color": "FF0000", "window": {"min": 0.0, "max": 125.0, "start": 2.0, "end": 110.0}},
                    {"label": "c2", "color": "0000FF", "window": {"min": 0.0, "max": 125.0, "start": 3.0, "end": 120.0}},
                ],
                "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
            },
        }
    )
    scene.create_array("0", shape=(2, 3, 2, 12, 12), dtype=np.uint8, chunks=(1, 1, 1, 12, 12), write_data=False)
    scene["0"][0, 0, 0, 0:2, 0:2] = np.array([[5, 7], [9, 11]], dtype=np.uint8)


def _lif_ome_xml() -> str:
    images: list[str] = []
    for image_index in range(17):
        if image_index == 0:
            images.append('<Image Name="Graph Scene"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="4" SizeX="4"/></Image>')
            continue
        if image_index == 14:
            name = "E8Flex/Stripes/300_300"
        elif image_index == 15:
            name = "E8Flex/Stripes/300_300_Merged"
        elif image_index == 16:
            name = "E8Flex/Stripes/300_300"
        else:
            name = f"Placeholder {image_index}"
        if image_index == 15:
            planes = []
            for t_index in range(2):
                for c_index in range(3):
                    for z_index in range(2):
                        delta_t = (t_index * 10.0) + float(c_index) + (z_index * 0.25)
                        planes.append(
                            f'<Plane TheT="{t_index}" TheC="{c_index}" TheZ="{z_index}" DeltaT="{delta_t}" DeltaTUnit="s" PositionZ="{z_index * 2.0}" PositionZUnit="Âµm"/>'
                        )
            images.append(
                f'<Image Name="{name}"><Pixels SizeT="2" SizeC="3" SizeZ="2" SizeY="12" SizeX="12" PhysicalSizeZ="2.0" PhysicalSizeZUnit="Âµm">{"".join(planes)}</Pixels></Image>'
            )
        elif image_index in {14, 16}:
            images.append(
                f'<Image Name="{name}"><Pixels SizeT="2" SizeC="3" SizeZ="2" SizeY="12" SizeX="12" PhysicalSizeZ="2.0" PhysicalSizeZUnit="Âµm"/></Image>'
            )
        else:
            images.append(f'<Image Name="{name}"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="4" SizeX="4"/></Image>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'{"".join(images)}'
        "</OME>"
    )


def _vsi_ome_xml() -> str:
    planes: list[str] = []
    values: list[str] = []
    units: list[str] = []
    counter = 1
    for t_index in range(100):
        for z_index in range(34):
            planes.append(
                f'<Plane TheT="{t_index}" TheC="0" TheZ="{z_index}" PositionZ="{z_index * 0.75}" PositionZUnit="Âµm" PositionY="0.0" PositionYUnit="Âµm" PositionX="0.0" PositionXUnit="Âµm"/>'
            )
    for c_index in range(1):
        for t_index in range(100):
            for z_index in range(34):
                value = float(t_index * 10)
                values.append(
                    f'<XMLAnnotation ID="Annotation:V{counter}" Namespace="openmicroscopy.org/OriginalMetadata">'
                    f"<Value><OriginalMetadata><Key>C555 Value #{counter:04d}</Key><Value>{value}</Value></OriginalMetadata></Value>"
                    f"</XMLAnnotation>"
                )
                units.append(
                    f'<XMLAnnotation ID="Annotation:U{counter}" Namespace="openmicroscopy.org/OriginalMetadata">'
                    f"<Value><OriginalMetadata><Key>C555 Units #{counter:04d}</Key><Value>10^-3s^1</Value></OriginalMetadata></Value>"
                    f"</XMLAnnotation>"
                )
                counter += 1
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image Name="C555">'
        '<Pixels SizeT="100" SizeC="1" SizeZ="34" SizeY="32" SizeX="32">'
        f'{"".join(planes)}'
        "</Pixels>"
        "</Image>"
        f'<StructuredAnnotations>{"".join(values)}{"".join(units)}</StructuredAnnotations>'
        "</OME>"
    )
