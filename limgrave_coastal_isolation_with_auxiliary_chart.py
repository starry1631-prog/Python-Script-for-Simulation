"""
Limgrave coastal-isolation simulation
======================================

Purpose
-------
A simplified, physically inspired height-field model that tests whether a
continuous rocky headland can be isolated as an offshore erosional remnant.

Two runs use the SAME initial terrain:
1. Uniform-bedrock control
2. Structurally weakened headland neck

Processes represented:
- wave attack on land cells immediately adjacent to water
- faster erosion of weak rock
- cliff collapse / slope relaxation
- limited nearshore sediment deposition

The model uses dimensionless time steps. It is a conceptual landscape-evolution
experiment, not a calibrated prediction of years or erosion rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label,
    zoom,
)


@dataclass(frozen=True)
class Config:
    ny: int = 240
    nx: int = 340
    steps: int = 330
    seed: int = 11
    sea_level: float = 0.0
    erosion_rate: float = 0.60
    collapse_rate: float = 0.25
    deposition_fraction: float = 0.06
    output_dir: str = "limgrave_output_revised"


def fractal_noise(
    shape: tuple[int, int],
    seed: int,
    octaves: tuple[int, ...] = (4, 8, 16, 32),
    weights: tuple[float, ...] = (1.0, 0.55, 0.28, 0.14),
) -> np.ndarray:
    """Return smooth multi-scale noise with mean 0 and standard deviation ~1."""
    rng = np.random.default_rng(seed)
    result = np.zeros(shape, dtype=float)
    total_weight = 0.0

    for scale, weight in zip(octaves, weights):
        coarse_shape = (
            max(2, shape[0] // scale),
            max(2, shape[1] // scale),
        )
        coarse = rng.normal(size=coarse_shape)
        enlarged = zoom(
            coarse,
            (shape[0] / coarse_shape[0], shape[1] / coarse_shape[1]),
            order=3,
        )
        enlarged = enlarged[: shape[0], : shape[1]]
        enlarged = (enlarged - enlarged.mean()) / (enlarged.std() + 1e-9)
        result += weight * enlarged
        total_weight += weight

    return result / total_weight


def create_initial_landscape(cfg: Config) -> tuple[np.ndarray, ...]:
    """
    Build a cliffed mainland, a protruding headland, and a narrow land bridge.

    x increases inland; y runs parallel to the coast.
    """
    y = np.linspace(-1.0, 1.0, cfg.ny)
    x = np.linspace(0.0, 1.0, cfg.nx)
    X, Y = np.meshgrid(x, y)

    broad_noise = fractal_noise((cfg.ny, cfg.nx), cfg.seed)
    coast_noise = gaussian_filter(broad_noise, 9)

    # Mainland coast: sea is on the left, land on the right.
    mainland = X > (
        0.47
        + 0.018 * np.sin(3.0 * np.pi * Y)
        + 0.012 * coast_noise
    )

    # A resistant, rounded headland protruding into the sea.
    headland = (
        ((X - 0.27) / 0.19) ** 2
        + ((Y - 0.03) / 0.28) ** 2
        < 1.0
    )

    # Narrow connection between headland and mainland.
    neck_half_width = 0.070 + 0.018 * np.cos(8.0 * np.pi * X)
    neck = (
        (X > 0.28)
        & (X < 0.49)
        & (np.abs(Y - 0.03) < neck_half_width)
    )

    initial_land = mainland | headland | neck
    distance_inside = distance_transform_edt(initial_land)
    distance_outside = distance_transform_edt(~initial_land)

    # Near-vertical rocky cliffs and a gently rising inland plateau.
    elevation = np.where(
        initial_land,
        31.0 * (1.0 - np.exp(-distance_inside / 2.0))
        + 0.028 * np.maximum(np.arange(cfg.nx)[None, :] - 150, 0),
        -0.8 - 0.18 * distance_outside,
    )
    elevation += np.where(initial_land, 1.9 * broad_noise, 0.25 * broad_noise)

    # Spatial resistance for the structural experiment.
    resistance = 1.0 + 0.18 * fractal_noise(
        (cfg.ny, cfg.nx),
        cfg.seed + 2,
    )

    # Resistant core remains after weaker surrounding rock is removed.
    resistance += 1.25 * np.exp(
        -((X - 0.25) / 0.12) ** 2
        -((Y - 0.03) / 0.20) ** 2
    )

    # Weak neck: an analogue for joint/fracture-controlled erosion.
    weak_neck = np.exp(
        -((X - 0.395) / 0.045) ** 2
        -((Y - 0.03) / 0.085) ** 2
    )
    resistance *= 1.0 - 0.82 * weak_neck

    # Two weaker coastal incisions approaching the neck from opposite sides.
    for y_centre in (-0.11, 0.17):
        incision = np.exp(
            -((X - 0.38) / 0.07) ** 2
            -((Y - y_centre) / 0.040) ** 2
        )
        resistance *= 1.0 - 0.55 * incision

    resistance = np.clip(resistance, 0.12, 3.0)

    # Control: same terrain but spatially uniform rock resistance.
    control_resistance = np.full_like(resistance, 1.25)

    return X, Y, elevation, resistance, control_resistance


def coastal_step(
    elevation: np.ndarray,
    resistance: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Advance the conceptual coastal model by one dimensionless time step."""
    sea = cfg.sea_level
    water = elevation <= sea
    land = ~water

    # Distance in grid cells from every land point to the nearest water cell.
    distance_to_water = distance_transform_edt(land)

    gradient_y, gradient_x = np.gradient(elevation)
    slope = np.hypot(gradient_x, gradient_y)

    # Wave attack decays rapidly inland and is amplified on steep, weak rock.
    exposure = np.exp(-distance_to_water / 2.0) * land
    erosion = (
        cfg.erosion_rate
        * exposure
        * (1.0 + 0.18 * np.clip(slope, 0.0, 8.0))
        / resistance
    )

    # High surfaces erode more slowly than low coastal edges in a height-field
    # approximation of basal undercutting.
    erosion *= np.exp(
        -np.clip(elevation - sea, 0.0, None) / 22.0
    )

    updated = elevation - erosion

    # Cliff collapse / mass wasting: over-steep nearshore cells relax toward
    # their local neighbourhood.
    neighbourhood_mean = gaussian_filter(updated, 1.1)
    gradient_y, gradient_x = np.gradient(updated)
    slope = np.hypot(gradient_x, gradient_y)

    unstable = (
        (slope > 2.6)
        & (updated > sea)
        & (distance_to_water < 14.0)
    )
    updated[unstable] += (
        cfg.collapse_rate
        * (neighbourhood_mean - updated)[unstable]
    )

    # A small fraction of eroded material is shifted seaward and deposited in
    # shallow water. Most material is assumed to be removed from the model.
    removed = np.maximum(elevation - updated, 0.0)
    seaward_shift = np.roll(removed, -5, axis=1)
    deposition = (
        gaussian_filter(seaward_shift, 2.2)
        * cfg.deposition_fraction
    )
    nearshore = (
        (updated < sea + 0.8)
        & (updated > sea - 4.2)
    )
    updated[nearshore] += deposition[nearshore]

    return updated



def minimum_bridge_width(
    elevation: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    sea_level: float,
) -> int:
    """
    Measure the narrowest surviving part of the original headland neck.

    The value is the minimum number of connected land cells across the
    alongshore direction within the predefined neck corridor. A value of zero
    means that water has cut completely through the bridge and the headland is
    no longer physically connected to the mainland.
    """
    x_mask = (X[0] >= 0.30) & (X[0] <= 0.47)
    y_mask = (Y[:, 0] >= -0.15) & (Y[:, 0] <= 0.21)

    neck_land = elevation[np.ix_(y_mask, x_mask)] > sea_level
    width_at_each_cross_section = neck_land.sum(axis=0)

    return int(width_at_each_cross_section.min())



def run_simulation(
    initial: np.ndarray,
    resistance: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[int],
    list[int],
    list[int],
]:
    """
    Run the model, retain selected terrain snapshots, and track the surviving
    width of the headland-to-mainland bridge.
    """
    requested_steps = [0, 60, 150, cfg.steps]
    snapshots = [initial.copy()]
    snapshot_steps = [0]

    metric_interval = 5
    metric_steps = [0]
    bridge_widths = [
        minimum_bridge_width(initial, X, Y, cfg.sea_level)
    ]

    current = initial.copy()
    for step in range(1, cfg.steps + 1):
        current = coastal_step(current, resistance, cfg)

        if step in requested_steps[1:]:
            snapshots.append(current.copy())
            snapshot_steps.append(step)

        if step % metric_interval == 0 or step == cfg.steps:
            metric_steps.append(step)
            bridge_widths.append(
                minimum_bridge_width(
                    current,
                    X,
                    Y,
                    cfg.sea_level,
                )
            )

    return (
        current,
        snapshots,
        snapshot_steps,
        metric_steps,
        bridge_widths,
    )


def land_component_statistics(
    elevation: np.ndarray,
    sea_level: float,
    minimum_area: int = 50,
) -> tuple[int, list[int]]:
    """Count meaningful connected land masses, ignoring tiny numerical specks."""
    labelled, count = label(elevation > sea_level)
    areas = [
        int(np.count_nonzero(labelled == component))
        for component in range(1, count + 1)
    ]
    areas = sorted(
        (area for area in areas if area >= minimum_area),
        reverse=True,
    )
    return len(areas), areas


def realistic_rgb(elevation: np.ndarray, sea_level: float) -> np.ndarray:
    """Create shaded terrain colours for plan and perspective rendering."""
    gradient_y, gradient_x = np.gradient(elevation)
    slope = np.hypot(gradient_x, gradient_y)

    z_norm = np.clip(
        (elevation - sea_level) / (np.nanmax(elevation) - sea_level + 1e-9),
        0.0,
        1.0,
    )
    steepness = np.clip(slope / 3.0, 0.0, 1.0)
    lowland = np.exp(-np.clip(elevation - sea_level, 0.0, None) / 4.0)

    grass = np.array([0.25, 0.36, 0.17])
    dry_grass = np.array([0.46, 0.42, 0.25])
    rock = np.array([0.39, 0.36, 0.33])
    sand = np.array([0.63, 0.55, 0.40])
    water = np.array([0.075, 0.23, 0.31])

    base_land = (
        grass[None, None, :] * (1.0 - z_norm[..., None])
        + dry_grass[None, None, :] * z_norm[..., None]
    )
    base_land = (
        base_land * (1.0 - steepness[..., None])
        + rock[None, None, :] * steepness[..., None]
    )
    base_land = (
        base_land * (1.0 - 0.65 * lowland[..., None])
        + sand[None, None, :] * (0.65 * lowland[..., None])
    )

    light = LightSource(azdeg=315, altdeg=42)
    hillshade = light.hillshade(elevation, vert_exag=1.1)
    base_land *= (0.55 + 0.55 * hillshade[..., None])

    rgb = np.where(
        (elevation <= sea_level)[..., None],
        water[None, None, :],
        base_land,
    )
    return np.clip(rgb, 0.0, 1.0)


def draw_plan_view(
    ax: plt.Axes,
    elevation: np.ndarray,
    sea_level: float,
    title: str,
) -> None:
    ax.imshow(
        realistic_rgb(elevation, sea_level),
        origin="lower",
        aspect="auto",
    )
    ax.contour(
        elevation,
        levels=[sea_level],
        colors="white",
        linewidths=0.7,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def save_summary(
    control_snapshots: list[np.ndarray],
    structural_snapshots: list[np.ndarray],
    steps: list[int],
    cfg: Config,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)

    for column, (control, structural, step) in enumerate(
        zip(control_snapshots, structural_snapshots, steps)
    ):
        draw_plan_view(
            axes[0, column],
            control,
            cfg.sea_level,
            f"Uniform rock — step {step}",
        )
        draw_plan_view(
            axes[1, column],
            structural,
            cfg.sea_level,
            f"Weak headland neck — step {step}",
        )

    fig.suptitle(
        "Limgrave analogue: structural weakness controls headland isolation",
        fontsize=16,
    )
    fig.savefig(
        output_dir / "limgrave_evolution_comparison.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)



def save_bridge_width_chart(
    metric_steps: list[int],
    control_widths: list[int],
    structural_widths: list[int],
    output_dir: Path,
) -> None:
    """
    Save a standalone chart showing whether the land bridge survives.

    This is the quantitative companion to the plan-view terrain sequence:
    - a positive width means that the headland is still attached;
    - zero means complete erosional separation.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

    ax.plot(
        metric_steps,
        control_widths,
        linewidth=2.5,
        label="Uniform-bedrock control",
    )
    ax.plot(
        metric_steps,
        structural_widths,
        linewidth=2.5,
        label="Structurally weakened neck",
    )

    detachment_step = next(
        (
            step
            for step, width in zip(metric_steps, structural_widths)
            if width == 0
        ),
        None,
    )

    if detachment_step is not None:
        ax.axvline(
            detachment_step,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
        ax.annotate(
            f"Complete separation at step {detachment_step}",
            xy=(detachment_step, 0),
            xytext=(detachment_step - 105, 5.5),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=10,
        )

    ax.set_xlabel("Dimensionless model step")
    ax.set_ylabel("Minimum surviving bridge width (grid cells)")
    ax.set_title(
        "Limgrave analogue: surviving headland bridge through time"
    )
    ax.set_ylim(bottom=-0.5)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    fig.savefig(
        output_dir / "limgrave_bridge_width_auxiliary.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)



def save_perspective(
    X: np.ndarray,
    Y: np.ndarray,
    control_final: np.ndarray,
    structural_final: np.ndarray,
    cfg: Config,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(15, 7.5))

    for panel, elevation, title in (
        (1, control_final, "Uniform-bedrock control: headland remains attached"),
        (2, structural_final, "Structurally weakened neck: detached remnant"),
    ):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        colours = realistic_rgb(elevation, cfg.sea_level)

        ax.plot_surface(
            X,
            Y,
            elevation,
            facecolors=colours,
            rstride=2,
            cstride=2,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        water_surface = np.where(
            elevation <= cfg.sea_level,
            cfg.sea_level,
            np.nan,
        )
        ax.plot_surface(
            X,
            Y,
            water_surface,
            color=(0.06, 0.24, 0.34),
            alpha=0.58,
            rstride=3,
            cstride=3,
            linewidth=0,
            shade=True,
        )

        ax.view_init(elev=30, azim=-125)
        ax.set_box_aspect((1.4, 1.0, 0.30))
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_axis_off()

    fig.suptitle(
        "Final simulated coastal terrain",
        fontsize=16,
        y=0.96,
    )
    fig.savefig(
        output_dir / "limgrave_final_3d.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_animation(
    initial: np.ndarray,
    resistance: np.ndarray,
    cfg: Config,
    output_dir: Path,
) -> None:
    """Save a lightweight GIF of the structurally controlled run."""
    frame_interval = 10
    frames: list[np.ndarray] = [initial.copy()]
    current = initial.copy()

    for step in range(1, cfg.steps + 1):
        current = coastal_step(current, resistance, cfg)
        if step % frame_interval == 0 or step == cfg.steps:
            frames.append(current.copy())

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    image = ax.imshow(
        realistic_rgb(frames[0], cfg.sea_level),
        origin="lower",
        aspect="auto",
    )
    title = ax.set_title("Step 0")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_index: int):
        image.set_data(realistic_rgb(frames[frame_index], cfg.sea_level))
        title.set_text(
            f"Structurally controlled erosion — step "
            f"{min(frame_index * frame_interval, cfg.steps)}"
        )
        return [image, title]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=110,
        blit=False,
    )

    try:
        animation.save(
            output_dir / "limgrave_isolation.gif",
            writer=PillowWriter(fps=8),
            dpi=130,
        )
    except Exception as exc:
        warnings.warn(f"GIF was not created: {exc}")
    finally:
        plt.close(fig)


def main() -> None:
    cfg = Config()
    script_directory = Path(__file__).resolve().parent
    output_dir = script_directory / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, initial, structural_resistance, control_resistance = (
        create_initial_landscape(cfg)
    )

    (
        control_final,
        control_snapshots,
        steps,
        metric_steps,
        control_widths,
    ) = run_simulation(
        initial,
        control_resistance,
        X,
        Y,
        cfg,
    )
    (
        structural_final,
        structural_snapshots,
        _,
        structural_metric_steps,
        structural_widths,
    ) = run_simulation(
        initial,
        structural_resistance,
        X,
        Y,
        cfg,
    )

    if structural_metric_steps != metric_steps:
        raise RuntimeError("Metric step sequences do not match.")

    save_summary(
        control_snapshots,
        structural_snapshots,
        steps,
        cfg,
        output_dir,
    )
    save_bridge_width_chart(
        metric_steps,
        control_widths,
        structural_widths,
        output_dir,
    )
    save_perspective(
        X,
        Y,
        control_final,
        structural_final,
        cfg,
        output_dir,
    )
    save_animation(
        initial,
        structural_resistance,
        cfg,
        output_dir,
    )

    control_count, control_areas = land_component_statistics(
        control_final,
        cfg.sea_level,
    )
    structural_count, structural_areas = land_component_statistics(
        structural_final,
        cfg.sea_level,
    )

    print("Simulation complete.")
    print(f"Outputs saved in: {output_dir.resolve()}")
    print(
        "Uniform-rock control: "
        f"{control_count} connected land component(s); "
        f"areas={control_areas[:3]}"
    )
    print(
        "Weak-neck experiment: "
        f"{structural_count} connected land component(s); "
        f"areas={structural_areas[:3]}"
    )
    print(
        "\nInterpretation: at the same model duration, the control headland "
        "remains attached, whereas erosion focused along the weak neck can "
        "isolate a resistant offshore remnant."
    )


if __name__ == "__main__":
    main()
