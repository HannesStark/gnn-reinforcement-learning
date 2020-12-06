import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecVideoRecorder

from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


def record_video(env_id:str="CartPole-v1",
                 algo:str="ppo",
                 folder:str="rl-trained-agents",
                 video_folder:str="logs/videos/",
                 video_length:int=1000,
                 n_envs:int = 1,
                 deterministic:bool=False,
                 seed:int=0,
                 no_render:bool=False,
                 exp_id:int = 0 ):
    


    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={exp_id}")
    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    model_path = os.path.join(log_path, f"{env_id}.zip")

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path)


    is_atari = "NoFrameskip" in env_id

    env = create_test_env(
        env_id,
        n_envs=n_envs,
        stats_path=stats_path,
        seed=seed,
        log_dir=None,
        should_render=not no_render,
        hyperparams=hyperparams,
    )

    model = ALGOS[algo].load(model_path)

    obs = env.reset()

    # Note: apparently it renders by default
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"{algo}-{env_id}",
    )

    env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, _, _ = env.step(action)

    # Workaround for https://github.com/openai/gym/issues/893
    if n_envs == 1 and "Bullet" not in env_id and not is_atari:
        env = env.venv
        # DummyVecEnv
        while isinstance(env, VecEnvWrapper):
            env = env.venv
        if isinstance(env, DummyVecEnv):
            env.envs[0].env.close()
        else:
            env.close()
    else:
        # SubprocVecEnv
        env.close()
