import tyro
from nudge.evaluator import Evaluator


def main(
    env_name: str = "seaquest",
    agent_path: str = "out/runs/kangaroo_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.0_numenvs_60_steps_128_pretrained_False_joint_True_20",
    fps: int = 5,
    episodes: int = 2
    ) -> None:
    explainer = Evaluator(\
        episodes=episodes,
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        device="cuda:0",
        # env_kwargs=dict(render_oc_overlay=True),
        env_kwargs=dict(render_oc_overlay=False),
        render_predicate_probs=True)
    explainer.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)
