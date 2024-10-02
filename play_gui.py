import tyro
from nudge.renderer import Renderer

def main(
    env_name: str = "seaquest",
    agent_path: str = "out/runs/kangaroo_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.0_numenvs_60_steps_128_pretrained_False_joint_True_20",
    fps: int = 5,
    seed: int = 0
    ) -> None:
    renderer = Renderer(\
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=False),
        render_predicate_probs=True,
        seed = seed)
    renderer.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)
