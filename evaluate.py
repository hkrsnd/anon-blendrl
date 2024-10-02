import tyro
from nudge.evaluator import Evaluator
from nudge.evaluator_neuralppo import EvaluatorNeuralPPO

def main(
    env_name: str = "seaquest",
    agent_path: str = "out/runs/kangaroo_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.0_numenvs_60_steps_128_pretrained_False_joint_True_20",
    fps: int = 5,
    episodes: int = 2,
    model: str = 'blendrl',
    device: str = 'cpu'
    ) -> None:
    assert model in ['blendrl', 'neuralppo'], "Invalid model type; choose from ['blendrl', 'neuralppo']"
    if model == 'blendrl':
        evaluator = Evaluator(\
            episodes=episodes,
            agent_path=agent_path,
            env_name=env_name,
            fps=fps,
            deterministic=False,
            device=device,
            # env_kwargs=dict(render_oc_overlay=True),
            env_kwargs=dict(render_oc_overlay=False),
            render_predicate_probs=True)
        evaluator.run()
    elif model == 'neuralppo':
        evaluator = EvaluatorNeuralPPO(\
            episodes=episodes,
            agent_path=agent_path,
            env_name=env_name,
            fps=fps,
            deterministic=False,
            device=device,
            # env_kwargs=dict(render_oc_overlay=True),
            env_kwargs=dict(render_oc_overlay=False),
            render_predicate_probs=True)
    
    
if __name__ == "__main__":
    tyro.cli(main)
