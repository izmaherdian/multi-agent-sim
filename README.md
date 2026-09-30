<h1 align="center">Decentralized Formation Control for Swarm Quadcopters</h1>

<p align="center">
  <b>Improved Artificial Potential Field (IAPF) + Event-Based Reconfiguration Control (ERC)</b><br>
  A Python simulator of a 5-quadcopter swarm with full 6-DoF dynamics, cascaded PID control, obstacles and wind gusts.
</p>

<p align="center">
  <a href="https://doi.org/10.1007/s44444-026-00111-4"><img src="https://img.shields.io/badge/Paper-JKSU--ES%202026-0a66c2?logo=springer&logoColor=white" alt="Paper"></a>
  <a href="https://doi.org/10.1007/s44444-026-00111-4"><img src="https://img.shields.io/badge/DOI-10.1007%2Fs44444--026--00111--4-blue" alt="DOI"></a>
  <a href="https://youtube.com/playlist?list=PLDnYn768_7b8L7e5SkQXCjXeL2ngwJcWW"><img src="https://img.shields.io/badge/YouTube-Simulation%20Videos-red?logo=youtube&logoColor=white" alt="YouTube"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"></a>
</p>

<p align="center">
  <img src="docs/media/s2_narrow_erc.gif" width="560" alt="ERC swarm passing a narrow gap">
  <br><sub><i>ERC swarm shrinks its V-formation, switches to single-file tailgating to pass a 1 m gap, then re-forms.</i></sub>
</p>

---

## ✨ Highlights

- **Hierarchical and decentralized.** A swarm-level planner (IAPF / ERC) generates setpoints, and each drone runs its own cascaded PID loop.
- **Adaptive reconfiguration.** ERC estimates the free width ahead, **scales** the formation, and switches to **tailgating** when the gap is too narrow.
- **3D mission extensions.** Autonomous takeoff, 3D goal/waypoint navigation, circular-path tracking, and formation orientation along the heading.
- **High-fidelity plant.** Quaternion 6-DoF model derived with Kane's method (DJI F450 parameters), 2nd-order motor dynamics, and a periodic gust wind model.

## 🎬 Demo

| Scenario | IAPF (static) | ERC (adaptive, proposed) |
|:--|:--:|:--:|
| **1 · Basic obstacle avoidance** | <img src="docs/media/s1_basic_iapf.gif" width="330"> | <img src="docs/media/s1_basic_erc.gif" width="330"> |
| **2 · Narrow gap passage** | <img src="docs/media/s2_narrow_iapf.gif" width="330"><br>❌ stalls at the entrance · [▶ YouTube](https://youtu.be/ICVNZTQ32nY) | <img src="docs/media/s2_narrow_erc.gif" width="330"><br>✅ scale → tailgate → re-form · [▶ YouTube](https://youtu.be/dLnhyqWmDu0) |

| 3 · Narrow gap + wind gust | 4 · U-shaped trap (limitation) |
|:--:|:--:|
| <img src="docs/media/s3_wind_erc.gif" width="330"><br>[▶ V-shape](https://youtu.be/KQ1U5d6qmrI) · [▶ Polygon](https://youtu.be/we8lNgnma8Q) | <img src="docs/media/s4_utrap_erc.gif" width="330"><br>Local minimum of a purely reactive planner |
| **5 · Waypoint navigation (gust)** | **6 · Circular path tracking (gust)** |
| <img src="docs/media/s5_waypoint_erc.gif" width="330"><br>[▶ YouTube](https://youtu.be/vGlRz3ikaZE) | <img src="docs/media/s6_circular_erc.gif" width="330"><br>[▶ YouTube](https://youtu.be/hIcMqDXWrwI) |

<p align="center">
  <a href="https://youtube.com/playlist?list=PLDnYn768_7b8L7e5SkQXCjXeL2ngwJcWW">
    <img src="https://img.youtube.com/vi/dLnhyqWmDu0/hqdefault.jpg" width="420" alt="Watch the full playlist on YouTube">
  </a>
  <br><b><a href="https://youtube.com/playlist?list=PLDnYn768_7b8L7e5SkQXCjXeL2ngwJcWW">▶ Watch the full simulation playlist on YouTube</a></b>
</p>

## 🧠 Method at a Glance

<table>
<tr>
<td width="50%" align="center"><img src="docs/media/architecture.png"><br><sub><b>System architecture</b>: high-level planner + per-agent PID</sub></td>
<td width="50%" align="center"><img src="docs/media/high_level_planner.png"><br><sub><b>High-level planner</b>: state machine, behaviors, event trigger</sub></td>
</tr>
</table>

**Behaviors.** Every agent sums simple velocity behaviors. ERC blends *formation* and *tailgating* with a switching function σᵢ:

$$\mathbf{v}_{ref}^{ERC} = \mathbf{v}^{mig} + \mathbf{v}^{obs} + \mathbf{v}^{col} + \sigma_i\,\mathbf{v}^{form} + (1-\sigma_i)\,\mathbf{v}^{tail}$$

<table>
<tr>
<td align="center"><img src="docs/media/behavior_formation.png" width="180"><br><sub>Formation</sub></td>
<td align="center"><img src="docs/media/behavior_migration.png" width="180"><br><sub>Migration</sub></td>
<td align="center"><img src="docs/media/behavior_avoidance.png" width="180"><br><sub>Avoidance</sub></td>
<td align="center"><img src="docs/media/behavior_tailgating.png" width="180"><br><sub>Tailgating</sub></td>
<td align="center"><img src="docs/media/behavior_takeoff.png" width="180"><br><sub>Takeoff</sub></td>
</tr>
</table>

<table>
<tr>
<td width="33%" align="center"><img src="docs/media/v_shape_topology.png"><br><sub>V-shape topology</sub></td>
<td width="33%" align="center"><img src="docs/media/polygon_topology.png"><br><sub>Polygon topology</sub></td>
<td width="33%" align="center"><img src="docs/media/width_estimation.png"><br><sub>Free-width estimation w<sub>e</sub> that triggers scaling/tailgating</sub></td>
</tr>
</table>

➡️ Dynamics model, controller gains, full behavior equations and the hardware plan are in **[docs/TECHNICAL.md](docs/TECHNICAL.md)**.

## 📊 Key Results

| Scenario | Planner | Outcome | Time (s) | Avg. speed (m/s) | RMSE<sub>total</sub> (m) | Φ (order) |
|:--|:--|:--:|--:|--:|--:|--:|
| 1 · Basic obstacles | IAPF | ✅ | 66.790 | 0.492 | 0.236 | 0.892 |
| | **ERC** | ✅ | **65.745** | **0.503** | 0.241 | 0.890 |
| 2 · Narrow gap | IAPF | ❌ | 54.195 | 0.365 | 0.525 | 0.868 |
| | **ERC** | ✅ | **59.735** | **0.561** | 0.692 | **0.878** |
| 3 · Narrow gap + gust | **ERC** | ✅ | 59.735 | 0.557 | 0.694 | **0.925** |

<table>
<tr>
<td width="50%" align="center"><img src="docs/media/iapf_narrow_fail.png"><br><sub>IAPF: rigid formation gets stuck at the gap</sub></td>
<td width="50%" align="center"><img src="docs/media/erc_narrow_traj.png"><br><sub>ERC: formation → tailgating → formation</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/media/erc_mode_switch.png"><br><sub>ERC mode switching and formation scale factor</sub></td>
<td align="center"><img src="docs/media/erc_wind_effort.png"><br><sub>Control effort with vs. without gust</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/media/waypoint_traj.png" width="300"><br><sub>Waypoint navigation</sub></td>
<td align="center"><img src="docs/media/circle_traj.png"><br><sub>Circular path tracking</sub></td>
</tr>
</table>

## 🚀 Quick Start

```bash
git clone https://github.com/izmaherdian/multi-agent-sim.git
cd multi-agent-sim
pip install numpy==1.26.4 scipy==1.15.2 sympy==1.13.3 matplotlib==3.9.4
python main.py
```

Configure the run in [`agent/config.py`](agent/config.py):

| Setting | Options |
|:--|:--|
| `CONTROLLER` | `'erc'`, `'iapf'` |
| `OBSTACLE_SCHEME` | `'scheme1'` basic · `'scheme2'` narrow gap · `'scheme3'` U-trap · `'NONE'` |
| `WIND_TYPE` | `'NONE'`, `'FIXED'`, `'GUST'` |
| `PATH_TYPE` | `'goal'`, `'multi-goal'` (waypoints), `'circular'` |

At the bottom of [`main.py`](main.py), switch the call to `main_multi_agent()` for the swarm (the default `main()` runs a single quadcopter). Logs (`*.pkl`) and videos (`*.mp4`) are saved to `results/data/` and `results/videos/` (git-ignored). Post-process them with the scripts in [`visualization/`](visualization/), e.g. `plot_paper_figures.py`.

## 📁 Repository Structure

```
├── agent/            quadcopter model, cascaded PID, IAPF & ERC planners, config
├── environment/      obstacles, wind model, 3D animation canvas
├── visualization/    analysis & plotting scripts
├── main.py           simulation entry point
├── scratch/          early prototype scripts (not needed to run the simulator)
├── paper/latex/      LaTeX source + figures of the journal paper
└── docs/
    ├── paper/        published paper (PDF) + citation (.ris)
    ├── thesis/       undergraduate thesis report & slides (Bahasa Indonesia)
    ├── media/        GIFs and figures used in this README
    └── TECHNICAL.md  detailed model, controller and algorithm description
```

## 📄 Publication & Citation

📑 **[Read the paper (PDF)](docs/paper/herdian2026_jksu-es.pdf)** · [Publisher page](https://doi.org/10.1007/s44444-026-00111-4) · [LaTeX source](paper/latex/) · [Thesis report](docs/thesis/)

> I. A. Herdian, E. Ekawati, F. Mukhlish, P. Prabaswara, "Decentralized formation control system design for swarm quadcopters using an improved artificial potential field and event-based reconfiguration control," *Journal of King Saud University – Engineering Sciences*, 38(5), 41 (2026).

```bibtex
@article{Herdian2026,
  author  = {Herdian, Izma Alhazmi and Ekawati, Estiyanti and Mukhlish, Faqihza and Prabaswara, Pramoda},
  title   = {Decentralized formation control system design for swarm quadcopters using an improved artificial potential field and event-based reconfiguration control},
  journal = {Journal of King Saud University -- Engineering Sciences},
  volume  = {38},
  number  = {5},
  pages   = {41},
  year    = {2026},
  doi     = {10.1007/s44444-026-00111-4}
}
```

A RIS file is available at [`docs/paper/herdian2026_citation.ris`](docs/paper/herdian2026_citation.ris).

## 📜 License

The source code is released under the [MIT License](LICENSE). The paper, thesis and figures in `docs/` and `paper/` remain under their respective copyrights (the published paper is open access under its publisher's license).

<p align="center"><sub>Engineering Physics, Institut Teknologi Bandung</sub></p>
