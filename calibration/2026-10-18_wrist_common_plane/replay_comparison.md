# EP00 current calibration rendering comparison

216 frames, 15 fps, 14.4 s. Reference video remains unchanged.

- `outputs/replay_cl/ep00_wrist_real_old_new_calib_20261018.mp4`: real wrist / previous simulation cropped from reference / current candidate simulation.
- `outputs/replay_cl/ep00_real_vs_sim_2cam_calib_20261018.mp4`: real and current candidate, original two-camera layout.
- `outputs/replay_cl/calib_20261018/wrist_comparison_contact_sheet.png`: frames 50, 70, 84, 100.

Candidate: `common_plane_fit_sag` selected by development f40 in `calibration/2026-10-18_wrist_common_plane/refined/report.json`. Base pose, gravity, sag factors and wrist extrinsics applied jointly. Camera intrinsics retained. Original 95 mm cup retained for reference consistency. Formal calibration files were not activated or overwritten.

Recorded body and gripper observations are the targets; source body observations were verified exactly against the dataset parquet. The camera follows the simulated articulation, so physics tracking error is retained. Reference gripper was action-driven; current gripper is observation-driven. Cup dynamics changes therefore cannot be attributed solely to camera calibration. Frame 1 maps to reference video index 0, with no time shift or lag correction.

## Geometry incompatibility discovered

With the standard 5 mm near plane the candidate wrist view is mostly black. A direct ray/triangle intersection on the stock link5 visual mesh finds its center viewing ray intersects the housing 16.228 mm ahead of the candidate optical center. The new estimated mount is approximately 13 mm behind the old mount. This is evidence of an incompatibility between the candidate optical pose and the stock mechanical mesh; it does not identify which is physically wrong.

The delivered diagnostic comparison uses a **40 mm wrist near plane**, explicitly labeled in the video. This removes nearby geometry from the image; it does not move the camera, change its intrinsics, or fix the incompatibility. Do not use the missing nearby housing/finger appearance to assess calibration accuracy. Standard 5 mm outputs remain in `inputs/sim_fit_calib_20261018`; diagnostic outputs are in `inputs/sim_fit_calib_20261018_clip40`.

The 10/18 candidate and 10/14 reference were recorded on different dates. This is a diagnostic visual comparison, not a passed absolute calibration validation. More marker captures are not requested for this replay.

Inputs and limitations: `render_manifest.json`. Output verification and physics tracking residuals: `comparison_manifest.json`. Logs: `render.log`, `render_clip40.log`.
