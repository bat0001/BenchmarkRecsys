import argparse

def get_config():
    parser = argparse.ArgumentParser(description="GFlowNet Recommendation Training Script")
    
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of the latent space")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of hidden layers")
    parser.add_argument("--subset_size", type=int, default=4, help="Size of the subset to select")
    parser.add_argument("--batch_size_train", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_iterations", "--iterations", type=int, default=100,
                         help="Number of training iterations / online visits")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                    choices=["CIFAR-10", "CIFAR-100", "FashionMNIST", "COCO", "AMAZON"],
                    help="Name of the dataset")
    parser.add_argument("--target_classes", type=str, default="airplane:2,truck:2",
                        help="Target classes (CIFAR) and their reward, format 'class:reward,class2:reward,...'")
    parser.add_argument("--target_classes_coco", type=str, default="person:1,skateboard:2,motorcycle:2",
                    help="Target classes for COCO + reward, format 'class:reward,class2:reward,...'")
    parser.add_argument("--num_epochs_vae", type=int, default=10, help="Number of epochs for VAE training")
    parser.add_argument("--initial_temp", type=float, default=25.0, help="Initial temperature for exploration")
    parser.add_argument("--final_temp", type=float, default=1.0, help="Final temperature after all iterations")
    parser.add_argument("--model_type", type=str, default="vae", choices=["vae", "convnext"],
                        help="Choose which model backbone to use: 'vae' or 'convnext'.")

    # weights & Biases 
    parser.add_argument("--wandb_project", type=str, default="BenchmarkRecSys", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="BenchmarkRecSys", help="Weights & Biases run name")
    parser.add_argument('--no_recommendation_logs', action='store_true',
                        help="Disable logging of image recommendations to WandB (to speed up logging)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # models
    parser.add_argument('--classical_gflownet', action='store_true', help="Train and evaluate the Classical GFlowNet.")
    parser.add_argument('--preference_binary_gflownet', action='store_true', help="Train and evaluate the Preference Learning GFlowNet.")
    parser.add_argument('--preference_dpo_gflownet', action='store_true', help="Train and evaluate the Preference Learning GFlowNet using dpo.")
    parser.add_argument('--all', action='store_true', help="Train and evaluate both GFlowNet models.")
    parser.add_argument('--random_baseline', action='store_true', help="Evaluate the Random Baseline as a performance comparison.")
    parser.add_argument('--MAB', action='store_true', help="Train and evaluate Multi-Armed Bandits.")

    # mode
    parser.add_argument('--it', action='store_true', help='Enable interactive mode with human feedback')


    parser.add_argument("--bandit_ckpt", type=str,
                    default="baselines/bandit/linucb.pt",
                    help="Path to bandit.pt file")


    #amazon tmp

    parser.add_argument("--amazon_path", type=str, default="data/amazon.csv",
                    help="Chemin vers le CSV Amazon Reviews")
    parser.add_argument(
        "--amazon_min_pos",
        type=int, default=1,
        help="Nombre minimum de rewards positifs pour qu’un item soit gardé"
    )
    parser.add_argument(
        "--amazon_subset",
        type=int,
        default=7000,
        help="If set, keep only this many rows from the Amazon CSV (random sample).",
    )



    # Bandits
    parser.add_argument('--max_items', action='store_true', default=100)
    parser.add_argument('--abtest', action='store_true')
    parser.add_argument('--ucb',    action='store_true')
    parser.add_argument('--abtest_n_test', type=int, default=1000)
    parser.add_argument('--ucb_c',  type=float, default=2.0)
    parser.add_argument('--thompson', action='store_true',
                    help="Activate Thompson Sampling baseline")
    parser.add_argument('--epsilon', type=float, default=0.1,
                    help="ε value for ε‑greedy")
    parser.add_argument('--epsilon_greedy', action='store_true',
                    help="Run ε‑greedy baseline")
    parser.add_argument("--simple_ts", action="store_true",
                    help="Run the minimal Thompson‑Sampling baseline")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    parser.add_argument("--simple_ucb", action="store_true",
                    help="Run the minimal UCB baseline")
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument("--num_visit", action="store_true", default=1000)
    # plots
    parser.add_argument(
        "--plots", action="store_true",
        help="WIP plots ..."
    )

    args, _ = parser.parse_known_args()
    
    target_classes = dict(item.split(":") for item in args.target_classes.split(","))
    target_classes = {k: int(v) for k, v in target_classes.items()}
    args.target_classes = target_classes

    return args

