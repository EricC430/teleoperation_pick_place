# ---------------------------------------------------------------------------
# Laptop (Windows) LeRobot setup — recording + inference deployment machine.
#
# Single source of truth for how the laptop environment is built. Same role as
# scripts/run_container.sh on the GPU box: docs explain *why*, this script is
# *what actually runs*.
#
# WHY THIS SCRIPT EXISTS
#   LeRobot's pyproject.toml applies the CUDA wheel index only on Linux:
#
#       [tool.uv.sources]
#       torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
#
#   On Windows that marker is false, uv falls back to PyPI, and PyPI's Windows
#   default is the CPU build. So a plain install silently yields torch+cpu.
#   That is LeRobot issue #4093 -- a hard-coded platform condition, not a
#   transient bug. We cannot fix it upstream and must not edit the cloned repo,
#   so we repair it after every install.
#
#   ORDER MATTERS. LeRobot's constraints (torch>=2.7,<2.12.0 and
#   torchvision>=0.22.0,<0.27.0) will overwrite anything pre-installed.
#   Install LeRobot first, repair torch second. Never the reverse.
#
#   See docs/environment.md for the full write-up.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# --- 1. LeRobot -------------------------------------------------------------
# Only the extras this machine actually needs:
#   core_scripts = dataset + hardware + viz  -> lerobot-record / replay / calibrate
#   feetech      = SO-ARM motor support
# Add "dynamixel" if the lab arm turns out to be OMX.
# Do NOT use [all]: it pulls placo -> pin -> coal-library -> cmeel-qhull, whose
# build backend shells out to tooling that does not exist on Windows and fails
# with WinError 2. It also drags in hundreds of packages we never use, each one
# a version-drift risk against the GPU box.
uv pip install -e ".\lerobot[core_scripts,feetech]"

# --- 2. Repair torch --------------------------------------------------------
# Versions must stay inside LeRobot's bounds or the next install will fight us.
# cu126 chosen for this laptop's driver; the GPU box uses cu128. The CUDA build
# variant does NOT need to match across machines -- only the torch version does,
# and LeRobot's <2.12.0 cap already pins both to 2.11.0.
uv pip install --force-reinstall "torch==2.11.0" "torchvision>=0.22.0,<0.27.0" `
    --index-url https://download.pytorch.org/whl/cu126

# --- 3. Fail loudly, now, not during inference ------------------------------
# A CPU-only torch does not error. Recording works fine (no model involved);
# the problem only appears later as inference too slow to close the control
# loop -- at which point the obvious suspect is the policy, not the install.
python -c @"
import torch, torchvision, lerobot
print(f'torch        {torch.__version__}')
print(f'cuda         {torch.cuda.is_available()}')
print(f'torchvision  {torchvision.__version__}')
print(f'lerobot      {lerobot.__version__}')
assert torch.cuda.is_available(), 'CUDA unavailable -- see docs/environment.md, laptop section'
assert lerobot.__version__ == '0.6.2', f'lerobot version drift: {lerobot.__version__} != 0.6.2 (GPU box)'
"@

Write-Host ""
Write-Host "OK. Record the exact versions in the version table in docs/environment.md." -ForegroundColor Green
