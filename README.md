# openarm_sim

Cảnh MuJoCo tái tạo setup OpenArm v1.0 hai tay ngoài đời: robot đứng trên mặt bàn, hộp kê phía trước làm
mặt gắp, camera Intel RealSense D435i gắn ở ngực (mặt trước cột chân đế). Dùng để thu dữ liệu (ảnh camera
ngực + trạng thái khớp) và test trên sim. Mọi số đo nằm trong `config/scene.yaml`.

Quy ước: x trước, y trái, z lên; **z = 0 là mặt bàn đặt chân đế**, gốc toạ độ = gốc chân đế robot.

## Cài đặt

```bash
cd ~/OpenArm_Sim
uv venv -p 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Render offscreen dùng EGL (`MUJOCO_GL=egl`, đặt sẵn trong `openarm_sim/render.py`). Máy không có GPU NVIDIA
vẫn chạy qua Mesa (đã thử trên Intel Arc tích hợp). OSMesa không cài trên máy này. Khi script thoát có thể in
ra `EGLError ... eglDestroyContext` trong `__del__`: lỗi này vô hại.

## Chạy

```bash
.venv/bin/python scripts/build_scene.py          # config/scene.yaml -> scenes/openarm_chest_cam.xml
.venv/bin/python scripts/check_scene.py          # in số đo, kiểm tra va chạm/tầm với, render -> outputs/check/
.venv/bin/python -m pytest -q                    # test hình học
.venv/bin/python -m mujoco.viewer --mjcf scenes/openarm_chest_cam.xml   # xem tương tác (cần màn hình)
```

`check_scene.py` ghi:
- `chest_color.png` (1280×720), `chest_depth_mm.png` (uint16, mm, giống định dạng RealSense),
  `chest_depth_vis.png` (tô màu), `chest_depth_rgb.png`;
- `overview.png`, `side.png`, `camera_closeup.png`.

Mã thoát khác 0 nếu có cảnh báo (va chạm với hộp/bàn/camera ở q = 0, hoặc tâm mặt hộp ngoài tầm với).

## Chỉnh số đo cho khớp ảnh thật

1. Chụp một ảnh màu từ D435i thật (đúng độ phân giải 1280×720 nếu được), robot ở tư thế bất kỳ không che hộp.
2. So ảnh, thử giá trị mới bằng `--set` (không sửa YAML):
   ```bash
   .venv/bin/python scripts/compare_real.py --real path/to/real.png --grid 4
   .venv/bin/python scripts/compare_real.py --real path/to/real.png --set camera.pitch_deg=43 --set box.gap_from_column=0.15
   ```
   Kết quả ở `outputs/compare/`: `side_by_side.png`, `blend.png` (chồng mờ, `--alpha`), `edges.png`
   (biên cạnh của ảnh sim màu đỏ trên ảnh thật, dễ so nhất).
3. Khi đã trùng, chép giá trị vào `config/scene.yaml`, chạy lại `build_scene.py`, `check_scene.py` và `pytest`.
   Lưu ý: test đang khoá các số đo của bản đầu (0.58 m, 45°, 0.16 m, 0.14 m). Nếu đo lại ra số khác thì sửa
   cả giá trị trong test.

Gợi ý: pitch và chiều cao camera làm biên hộp trượt lên/xuống trong ảnh; `offset_x_from_column` và
`box.gap_from_column` có tác dụng gần giống nhau (đều đổi khoảng cách camera–hộp). Nên cố định số nào đo được
chắc chắn bằng thước trước. Nếu ảnh thật bị crop (ví dụ 640×480), truyền `--fovy` bằng FOV dọc thật của ảnh đó.

### Các khoá chính trong `config/scene.yaml`

| Khoá | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `box.top_height` | 0.16 | mặt trên hộp so với mặt bàn |
| `box.gap_from_column` | 0.14 | mép gần hộp cách mặt trước cột (theo x) |
| `box.size_x`, `box.size_y`, `box.center_y` | 0.40, 0.60, 0 | kích thước, vị trí ngang của hộp |
| `camera.lens_center_height` | 0.58 | z của quang tâm (mặt phẳng zero-depth) |
| `camera.offset_x_from_column` | 0.03 | từ mặt trước cột tới quang tâm theo x |
| `camera.housing_center_y` | 0 | y của tâm vỏ camera |
| `camera.pitch_deg` / `yaw_deg` / `roll_deg` | 45 / 0 / 0 | hướng camera; pitch dương = chúc xuống |
| `camera.color` / `camera.depth` | 42°, 1280×720 / 58°, 848×480 | fovy dọc và độ phân giải |
| `robot.column_probe_z` | [0.30, 0.60] | dải z để đo mặt trước cột từ mesh |
| `box.rgba` | đen | màu hộp kê |
| `target_pad.size`, `offset_xy` | 0.12×0.12, [0, 0] | ô trắng (đích đặt vật), lệch so với tâm mặt hộp |
| `objects.mentos_tin.size` / `mass` | 0.060×0.040×0.018 / 0.045 | hộp kẹo Mentos sắt (**ước lượng**, cần đo lại) |
| `objects.mentos_tin.pos_xy`, `yaw_deg` | [0.27, −0.15], 0 | vị trí ban đầu trên mặt hộp |

Task thử: gắp `mentos_tin` (thân tự do, có freejoint) bỏ vào ô `target_pad`. Site `mentos_tin_center` và
`target_pad_center` (group 4) dùng để tính khoảng cách/điều kiện thành công.

## Mô hình hoá

- **Mặt trước cột** được đo từ mesh chân đế (`openarm_body_link0`): cắt lát ngang trong dải `column_probe_z`,
  lấy x lớn nhất. Kết quả x = 0.030 m (cột nhôm 60×60 mm). Trên z ≈ 0.613 là khối vai (`body_link0_5`), nhô ra
  tới x ≈ 0.065 ở z = 0.70. Vì vậy dải mặc định dừng ở 0.60 (dải 0.30–0.70 sẽ ra 0.065). Profile 0.30–0.70
  được in trong `check_scene.py`.
- **Camera** là body `chest_camera_link`, con của `openarm_body_link0` (đi theo robot). Khung theo quy ước
  `camera_link` của RealSense (x nhìn, y trái, z trên). Gốc nằm trên mặt phẳng quang tâm, ở tâm vỏ theo y.
  Offset lấy từ `realsense2_description/urdf/_d435.urdf.xacro`
  (IntelRealSense/realsense-ros, nhánh ros2-master, Apache-2.0):
  - depth (= infra1) ở y = +0.0175 so với tâm vỏ (`d435_cam_depth_py`);
  - color ở y = +0.015 so với depth (`d435_cam_depth_to_color_offset`), tức +0.0325 so với tâm vỏ (phía trái robot);
  - mặt kính ở phía trước quang tâm 4.3 mm (`d435_zero_depth_to_glass + d435_glass_to_front`).

  Camera MuJoCo `chest_color` và `chest_depth` nhìn theo −z của chúng, "lên" của ảnh là +y. Test kiểm tra
  phải-ảnh = phải-robot (−y thế giới). Có thêm hộp va chạm 90×25×25 mm quanh vỏ (group 3) để phát hiện tay
  đụng camera. Tấm đỡ màu đen nối cột với camera chỉ để hiển thị, không có va chạm.
- **Hộp** là geom box có va chạm (`work_box`). **Bàn** (`table_top`) có mặt trên ở z = 0.
- Keyframe `zero` là q = 0 (hai tay thả thẳng xuống).

## Cấu trúc

```
config/scene.yaml          mọi số đo
openarm_sim/scene.py       sinh MJCF (MjSpec) + các phép đo dùng chung
openarm_sim/render.py      render offscreen RGB/depth
scripts/build_scene.py     YAML -> scenes/openarm_chest_cam.xml
scripts/check_scene.py     in số đo, va chạm, tầm với, render ảnh
scripts/compare_real.py    so ảnh sim với ảnh D435i thật
scenes/                    MJCF sinh ra (đừng sửa tay)
tests/                     pytest
third_party/openarm_mujoco model gốc (xem SOURCE.md)
```

## Nguồn và license

- `third_party/openarm_mujoco/`: [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco), commit
  `e5b9e50`, Apache-2.0, © 2025 Enactic, Inc. Đã copy nguyên trạng `v1/` và `v0.3/meshes/d435.stl`, không sửa.
  Chi tiết trong `third_party/openarm_mujoco/SOURCE.md` và `LICENSE`.
- Offset quang học D435: [IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros)
  `realsense2_description/urdf/_d435.urdf.xacro` (Apache-2.0). Chỉ dùng các hằng số, không copy file.
- FOV: datasheet Intel RealSense D400 (color ≈ 69°×42°, depth ≈ 87°×58°).
