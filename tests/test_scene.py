import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from openarm_sim.scene import (BOX_GEOM, _geom_world_mesh, CAM_COLLISION_GEOM, TABLE_GEOM, body_front_profile,  # noqa: E402
                               box_top, camera_pose, load_config, load_scene, write_scene)

CFG = load_config(DEFAULT_CONFIG)
CAMS = [CFG["camera"]["color"]["name"], CFG["camera"]["depth"]["name"]]


@pytest.fixture(scope="session")
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("scene") / "scene.xml"
    write_scene(CFG, path)
    return load_scene(path)


@pytest.fixture(scope="session")
def column_front(scene):
    m, d = scene
    z0, z1 = CFG["robot"]["column_probe_z"]
    return max(x for _, x in body_front_profile(m, d, CFG["robot"]["base_body"], np.linspace(z0, z1, 7)))


def test_scene_loads(scene):
    m, _ = scene
    for name in CAMS:
        assert m.camera(name) is not None
    assert m.geom(BOX_GEOM) is not None


def test_generated_scene_file_loads():
    if not DEFAULT_SCENE.exists():
        pytest.skip("chưa chạy build_scene.py")
    mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))


def test_column_front_from_mesh(column_front):
    # cột chân đế OpenArm v1 rộng 60 mm, mặt trước ở x = 0.030
    assert column_front == pytest.approx(0.030, abs=0.002)


@pytest.mark.parametrize("cam", CAMS)
def test_camera_height(scene, cam):
    assert camera_pose(*scene, cam)["pos"][2] == pytest.approx(0.58, abs=0.005)


@pytest.mark.parametrize("cam", CAMS)
def test_camera_pitch(scene, cam):
    p = camera_pose(*scene, cam)
    assert p["pitch_deg"] == pytest.approx(45.0, abs=1.0)
    assert p["yaw_deg"] == pytest.approx(0.0, abs=0.5)
    assert p["roll_deg"] == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("cam", CAMS)
def test_camera_image_orientation(scene, cam):
    p = camera_pose(*scene, cam)
    assert p["forward"][0] > 0.5                      # nhìn ra trước
    assert p["up"][2] > 0.5                           # "lên" của ảnh hướng lên trời
    assert p["right"] == pytest.approx([0, -1, 0], abs=1e-6)  # phải của ảnh = phải của robot


def test_camera_centered_and_on_column(scene, column_front):
    m, d = scene
    color, depth = (camera_pose(m, d, c)["pos"] for c in CAMS)
    # RGB lệch về trái robot 15 mm so với depth (realsense2_description)
    assert color[1] - depth[1] == pytest.approx(0.015, abs=1e-6)
    # tâm hộp bao của mesh vỏ căn giữa y; bề rộng vỏ ~90 mm nằm ngang theo y
    v, _ = _geom_world_mesh(m, d, m.geom("chest_camera_visual").id)
    assert (v[:, 1].min() + v[:, 1].max()) / 2 == pytest.approx(0.0, abs=5e-4)
    assert v[:, 1].max() - v[:, 1].min() == pytest.approx(0.090, abs=1e-3)
    assert depth[0] - column_front == pytest.approx(CFG["camera"]["offset_x_from_column"], abs=1e-4)


def test_camera_attached_to_base(scene):
    m, _ = scene
    b = m.body("chest_camera_link")
    assert m.body(m.body_parentid[b.id]).name == CFG["robot"]["base_body"]


def test_box_top_height(scene):
    assert box_top(*scene)["z"] == pytest.approx(0.16, abs=0.002)


def test_column_box_gap(scene, column_front):
    assert box_top(*scene)["x_near"] - column_front == pytest.approx(0.14, abs=0.005)


def test_no_contact_with_env_at_zero(scene):
    m, d = scene
    env = {m.geom(n).id for n in (BOX_GEOM, TABLE_GEOM, CAM_COLLISION_GEOM)}
    bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]
           if c.geom1 in env or c.geom2 in env]
    assert not bad, bad
