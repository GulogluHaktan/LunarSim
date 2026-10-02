"""Real Apollo Lunar Module (LM) physical specifications.

Single source of truth for mass/propulsion/RCS numbers used by both
`lunarsim.rl.analytic_lander_env` and the Isaac-side USD asset builder
(`scripts/build_apollo_lm_asset.py`), so the two backends can never drift
apart on what "the vehicle" weighs or how hard it can push.

Two kinds of numbers live here, and they are labeled as such:

- Historical/documented values (mass breakdown, DPS thrust range and Isp,
  RCS jet thrust/count, overall vehicle height) come from published
  Apollo LM figures (NASA/Grumman program data, as commonly reported in
  Apollo mission press kits and LM systems handbooks).
- Geometry/inertia numbers marked "engineering estimate" are NOT measured
  values from the real vehicle -- they are a simplified solid-cylinder
  approximation used only so the sim has *a* physically consistent inertia
  tensor to integrate against. Replace them if a more accurate figure is
  ever sourced.
"""
from __future__ import annotations

from dataclasses import dataclass

G0 = 9.80665  # standard gravity, for Isp -> mass flow conversions


@dataclass(frozen=True)
class ApolloLMSpecs:
    # --- mass breakdown at Powered Descent Initiation (documented) ---
    total_mass_pdi_kg: float = 15103.0
    descent_propellant_kg: float = 8165.0
    # everything that isn't descent propellant: structure, ascent stage,
    # ascent propellant (untouched during the descent/landing phase)
    dry_mass_kg: float = total_mass_pdi_kg - descent_propellant_kg  # 6938.0

    # --- Descent Propulsion System, the single throttleable main engine (documented) ---
    dps_thrust_min_n: float = 4672.0     # ~1,050 lbf, minimum throttle setting
    dps_thrust_max_n: float = 45040.0    # ~10,125 lbf, max rated thrust
    dps_isp_s: float = 311.0
    dps_gimbal_max_deg: float = 6.0      # CG-trim gimbal; NOT used for attitude
                                          # control in this model -- see module
                                          # docstring in analytic_lander_env.

    # --- Reaction Control System: 4 quads x 4 thrusters (documented count/thrust) ---
    rcs_thruster_count: int = 16
    rcs_thruster_thrust_n: float = 445.0  # 100 lbf per jet, vacuum
    rcs_isp_s: float = 290.0
    rcs_propellant_kg: float = 287.0

    # --- geometry (documented) ---
    height_m: float = 7.04             # legs deployed, footpad to docking hatch
    footpad_span_m: float = 9.4        # diagonal span across deployed footpads
    footpad_radius_m: float = 0.47     # dish radius (commonly cited 37 in diameter)

    # --- geometry/inertia inputs (engineering estimate, see docstring) ---
    body_radius_m: float = 2.1          # structural core radius (not leg span)
    rcs_quad_radius_m: float = 1.78     # RCS quad mount radius from centerline
    rcs_jets_per_couple: int = 2        # jets fired together for a pure-rotation couple

    # --- landing gear (documented count; stroke is an engineering estimate) ---
    leg_count: int = 4
    leg_stroke_m: float = 0.81          # commonly cited primary-strut crush stroke (~32 in)


def leg_force_bounds_n(specs: "ApolloLMSpecs", mass_kg: float, safe_v_z_m_s: float, gravity_m_s2: float) -> tuple[float, float]:
    """Engineering-estimate per-leg touchdown force range (a, b) -- NOT a
    measured strut load rating. `a` is the static per-leg load at rest
    (weight/leg_count); `b` adds an energy-absorption estimate for
    touchdown at exactly the safe vertical-velocity limit, spreading the
    kinetic energy evenly over all legs' crush stroke.
    """
    weight_n = mass_kg * gravity_m_s2
    a = weight_n / specs.leg_count
    ke_j = 0.5 * mass_kg * safe_v_z_m_s ** 2
    b = a + ke_j / (specs.leg_count * specs.leg_stroke_m)
    return a, b


def hover_analysis(mass_kg: float, specs: "ApolloLMSpecs", gravity_m_s2: float = 1.62) -> dict:
    """What it takes to hover this vehicle at `mass_kg`, from the real DPS
    thrust range -- answers "hover için gerekli itki = k x g_ay" concretely:
    `k` is the thrust-to-weight ratio the DPS must hold at hover (always 1
    by definition; reported anyway for clarity), plus the throttle fraction
    that gives exactly that thrust and the resulting margin/authority.
    """
    weight_n = mass_kg * gravity_m_s2
    hover_thrust_n = weight_n  # by definition, k=1 at hover
    throttle_range_n = specs.dps_thrust_max_n - specs.dps_thrust_min_n
    hover_throttle_fraction = (hover_thrust_n - specs.dps_thrust_min_n) / throttle_range_n
    return {
        "mass_kg": mass_kg,
        "weight_n": weight_n,
        "hover_thrust_n": hover_thrust_n,
        "k_thrust_to_weight_at_hover": hover_thrust_n / weight_n,  # = 1.0
        "hover_throttle_fraction": hover_throttle_fraction,
        "max_thrust_to_weight": specs.dps_thrust_max_n / weight_n,
        "min_thrust_to_weight": specs.dps_thrust_min_n / weight_n,
        "can_hover": bool(specs.dps_thrust_min_n <= hover_thrust_n <= specs.dps_thrust_max_n),
    }


def moment_of_inertia(mass_kg: float, body_radius_m: float, height_m: float) -> tuple[float, float]:
    """Engineering-estimate moment of inertia for a solid cylinder of the
    given mass/radius/height -- NOT a measured Apollo LM inertia tensor.
    Returns (i_tilt, i_yaw): i_tilt is about a horizontal axis through the
    centroid (roll/pitch, i.e. tipping over); i_yaw is about the vertical
    axis (spin about the thrust axis).
    """
    i_tilt = mass_kg * (3.0 * body_radius_m ** 2 + height_m ** 2) / 12.0
    i_yaw = mass_kg * body_radius_m ** 2 / 2.0
    return i_tilt, i_yaw


def rcs_max_torque_n_m(specs: ApolloLMSpecs) -> float:
    """Max torque achievable about any one axis by firing one couple
    (`rcs_jets_per_couple` jets) at the quad radius. Same moment arm is
    used for tilt and yaw axes since all quads sit at the same radius."""
    return specs.rcs_thruster_thrust_n * specs.rcs_quad_radius_m * specs.rcs_jets_per_couple
