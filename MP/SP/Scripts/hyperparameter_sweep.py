import wandb
import tyro
import time

from training import train_agent

class Args:
    exp_name: str = "hyperparameter SP grid"
    """Experiment name"""
    
    wandb_project_name: str = "SP_grid"
    """the wandb's project name"""

    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    total_timesteps: int = 10000000
    """total timesteps of the experiments"""

    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""

    buffer_size: int = 1000000
    """the replay memory buffer size"""

    gamma: float = 0.99
    """the discount factor gamma"""

    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""

    batch_size: int = 32
    """the batch size of sample from the reply memory"""

    start_e: float = 1
    """the starting epsilon for exploration"""

    end_e: float = 0.01
    """the ending epsilon for exploration"""

    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""

    learning_starts: int = 80000
    """timestep to start learning"""

    train_frequency: int = 4
    """the frequency of training"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{int(time.time())}{args.adjust}"
    wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
                id = args.id,
                dir="/rdata/ong/Anuj/wandb"
            )
