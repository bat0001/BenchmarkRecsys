import argparse

def get_config():
    parser = argparse.ArgumentParser(description="GFlowNet Recommendation Training Script")
    
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of the latent space")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of hidden layers")
    parser.add_argument("--subset_size", type=int, default=4, help="Size of the subset to select")
    parser.add_argument("--batch_size_train", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_iterations", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                    choices=["CIFAR-10", "CIFAR-100", "FashionMNIST", "COCO"],
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

    args, _ = parser.parse_known_args()
    
    target_classes = dict(item.split(":") for item in args.target_classes.split(","))
    target_classes = {k: int(v) for k, v in target_classes.items()}
    args.target_classes = target_classes

    return args