import torch
from tensordict.nn import set_interaction_type, InteractionType
from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder


# TODO: saved video terminates prematurely despite increased max_steps param, needs fix
def save_as_mp4(env, max_steps, policy, scenario_name, logdir="saves", exp_name="vmas_logs", suffix="eval"):
    video_logger = CSVLogger(exp_name, logdir, video_format="mp4")
    print("Creating rendering env")
    env_with_render = TransformedEnv(env.base_env, env.transform.clone())
    env_with_render = env_with_render.append_transform(
        PixelRenderTransform(
            out_keys=["pixels"],
            # the np.ndarray has a negative stride and needs to be copied before being cast to a tensor
            preproc=lambda x: x.copy(),
            as_non_tensor=True,
            # asking for array rather than on-screen rendering
            mode="rgb_array",
        )
    )
    env_with_render = env_with_render.append_transform(
        VideoRecorder(logger=video_logger, tag=f"{scenario_name}")
    )
    with set_exploration_type(ExplorationType.MODE):
        print("Rendering rollout...")
        env_with_render.rollout(max_steps, policy=policy)
    print("Saving the video...")
    env_with_render.transform.dump(suffix=suffix)
    print("Saved! Saved directory tree:")
    video_logger.print_log_dir()


def save_as_gif(env, max_steps, policy, scenario_name):
    from PIL import Image
    def rendering_callback(env, td):
        env.frames.append(Image.fromarray(env.render(mode="rgb_array")))

    env.frames = []
    with torch.no_grad():
        env.rollout(
            max_steps=max_steps,
            policy=policy,
            callback=rendering_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
    print("saving gif... ")
    env.frames[0].save(
        f"{scenario_name}.gif",
        save_all=True,
        append_images=env.frames[1:],
        duration=3,
        loop=0,
    )
