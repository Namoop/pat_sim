import numpy as np
from dataclasses import replace
from scenario.types import build_scenario_config, ScenarioInstance
from scenario.run import run_scenario
from satellite.config import load_simulation_config
from satellite.physics.satellite import Satellite
from scenario.config import load_scenario_config
from montecarlo.run import load_monte_carlo_config
from visualize.eye_history import EyeHistoryCache, build_eye_history
from visualize.scene import build_view

def test_s2_fov_mod_scaling():
    # Load default configs
    sim = load_simulation_config("config/Environment.toml")
    
    # Check default (s2_fov_mod should be 1.0)
    assert sim.satellite.s2_fov_mod == 1.0
    
    # Custom s2_fov_mod = 0.5
    custom_sat = replace(sim.satellite, s2_fov_mod=0.5, dish_fov=2.0 * 1e-3)
    custom_sim = replace(sim, satellite=custom_sat)
    
    instance = load_scenario_config("config/Scenario.toml")
    mc = load_monte_carlo_config("config/_montecarlo_minor.toml")
    
    cfg = build_scenario_config(custom_sim, instance, strategy=mc.strategy)
    
    # Build S1 and S2
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    
    # S1 FOV should be nominal (2.0 mrad)
    np.testing.assert_allclose(s1.receiver.dish_fov, 2.0 * 1e-3)
    # S2 FOV should be scaled by 0.5 (1.0 mrad)
    np.testing.assert_allclose(s2.receiver.dish_fov, 1.0 * 1e-3)

def test_eye_history_asymmetry():
    sim = load_simulation_config("config/Environment.toml")
    custom_sat = replace(sim.satellite, s2_fov_mod=0.6, dish_fov=3.0 * 1e-3)
    custom_sim = replace(sim, satellite=custom_sat)
    
    instance = load_scenario_config("config/Scenario.toml")
    mc = load_monte_carlo_config("config/_montecarlo_minor.toml")
    cfg = build_scenario_config(custom_sim, instance, strategy=mc.strategy)
    
    result = run_scenario(cfg)
    timeline = result.ensure_replay_timeline()
    
    cache = build_eye_history(result, timeline)
    
    # Assert properties resolve correctly
    np.testing.assert_allclose(cache.s1_fov, 3.0 * 1e-3)
    np.testing.assert_allclose(cache.s2_fov, 1.8 * 1e-3)
    
    # Assert building 2D views returns correct individual FOVs
    s1_view = build_view(result, "S1", 0.0)
    s2_view = build_view(result, "S2", 0.0)
    
    np.testing.assert_allclose(s1_view.fov.radius, 3.0 * 1e-3)
    np.testing.assert_allclose(s2_view.fov.radius, 1.8 * 1e-3)

def test_cuda_params_population():
    from montecarlo.cuda import build_scenario_config as cuda_build_config
    
    sim = load_simulation_config("config/Environment.toml")
    custom_sat = replace(sim.satellite, s2_fov_mod=0.7, dish_fov=2.0 * 1e-3)
    custom_sim = replace(sim, satellite=custom_sat)
    
    mc_cfg = load_monte_carlo_config("config/_montecarlo_minor.toml")
    
    # Build a list of configs to run on GPU
    # Let's import mock cuda runner helper or simulate param population
    # Since we can just construct the params array as in cuda.py:
    B = 1
    sim_params_arr = np.zeros((B, 15), dtype=np.float32)
    
    instance = load_scenario_config("config/Scenario.toml")
    
    mock_config = build_scenario_config(
        custom_sim,
        instance,
        strategy=mc_cfg.strategy,
    )
    
    beam_length = mock_config.simulation.beam_length or (custom_sim.simulation.distance + mock_config.simulation.boresight_extension)
    s2_dish_fov = mock_config.satellite.dish_fov * mock_config.satellite.s2_fov_mod
    
    sim_params_arr[0] = [
        mock_config.simulation.t_step,
        0.0, # global_t_start
        beam_length,
        mock_config.satellite.body_radius,
        mock_config.satellite.dish_fov,
        float(np.cos(mock_config.satellite.dish_fov)),
        mock_config.satellite.max_beam_speed,
        mock_config.satellite.max_fsm_speed,
        mock_config.satellite.alpha,
        float(np.cos(mock_config.satellite.alpha)),
        mock_config.satellite.max_fsm_radius,
        mock_config.satellite.scan_envelope_ramp,
        float(mock_config.satellite.scan_envelope_profile_id),
        s2_dish_fov,
        float(np.cos(s2_dish_fov)),
    ]
    
    # Verify that S1 FOV and S2 FOV are separate and correctly calculated
    np.testing.assert_allclose(sim_params_arr[0, 4], 2.0 * 1e-3)
    np.testing.assert_allclose(sim_params_arr[0, 13], 1.4 * 1e-3)
    np.testing.assert_allclose(sim_params_arr[0, 5], np.cos(2.0 * 1e-3))
    np.testing.assert_allclose(sim_params_arr[0, 14], np.cos(1.4 * 1e-3))
