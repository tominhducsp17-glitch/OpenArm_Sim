import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from openarm_sim.scene import (BOX_GEOM, PAD_GEOM, _geom_world_mesh, arm_geoms, body_front_profile,  # noqa: E402
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


def test_no_arm_contact_with_env_at_zero(scene):
    m, d = scene
    arms = set(arm_geoms(m))
    bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]
           if (c.geom1 in arms) != (c.geom2 in arms)]
    assert not bad, bad


def test_box_is_black(scene):
    m, _ = scene
    assert max(m.geom_rgba[m.geom(BOX_GEOM).id][:3]) < 0.1


def test_target_pad(scene):
    m, d = scene
    g = m.geom(PAD_GEOM).id
    assert 2 * m.geom_size[g][:2] == pytest.approx([0.12, 0.12], abs=1e-6)
    assert min(m.geom_rgba[g][:3]) > 0.9                                  # trắng
    bt = box_top(m, d)
    assert d.geom_xpos[g][:2] == pytest.approx(bt["center"][:2], abs=1e-6)  # giữa mặt hộp
    assert d.geom_xpos[g][2] - m.geom_size[g][2] == pytest.approx(bt["z"], abs=1e-6)  # nằm trên mặt hộp


def test_mentos_tin(scene):
    m, d = scene
    o = CFG["objects"]["mentos_tin"]
    g = m.geom("mentos_tin_geom").id
    assert 2 * m.geom_size[g] == pytest.approx(o["size"], abs=1e-6)
    assert m.body_mass[m.body("mentos_tin").id] == pytest.approx(o["mass"], abs=1e-6)
    assert m.jnt_type[m.body_jntadr[m.body("mentos_tin").id]] == mujoco.mjtJoint.mjJNT_FREE
    # đáy hộp kẹo chạm mặt hộp kê, không chồng lên ô trắng lúc đầu
    assert d.geom_xpos[g][2] - o["size"][2] / 2 == pytest.approx(box_top(m, d)["z"], abs=1e-4)
    pad = m.geom(PAD_GEOM).id
    gap = np.abs(d.geom_xpos[g][:2] - d.geom_xpos[pad][:2]) - m.geom_size[pad][:2] - np.array(o["size"][:2]) / 2
    assert gap.max() > 0


def test_mentos_tin_rests(scene):
    m, _ = scene
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("zero").id)
    b = m.body("mentos_tin").id
    mujoco.mj_forward(m, d)
    p0 = d.xpos[b].copy()
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.xpos[b] - p0) < 0.002
