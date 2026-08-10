env = ManagerBasedRLEnv(
    cfg=G1RoughEnv3_3PlaygroundCfg()
)

while simulation_app.is_running():
    action = bridge.get_action().to(env.device)
    observations, reward, terminated, truncated, info = env.step(action)
    bridge.publish_observations(env, observations)
