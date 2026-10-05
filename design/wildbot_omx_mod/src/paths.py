"""Path config. BASE = design/wildbot_omx_mod ; REPO = repo root.
OMX meshes are not in the repo (URDF uses package://). Clone ROBOTIS-GIT/open_manipulator and point OMX_MESH_DIR at
open_manipulator_description/meshes/omx_f, or set env WB_OMX_MESH_DIR."""
import os
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
WILDBOT_DIR = Path(os.environ.get('WB_WILDBOT_DIR', REPO / 'assets' / 'wildbot_car_urdf'))
OMX_URDF = Path(os.environ.get('WB_OMX_URDF', REPO / 'assets' / 'omx_f' / 'omx_f.urdf'))
OMX_MESH_DIR = Path(os.environ.get('WB_OMX_MESH_DIR', BASE / 'third_party' / 'open_manipulator' / 'open_manipulator_description' / 'meshes' / 'omx_f'))
