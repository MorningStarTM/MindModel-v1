config = {
    'state_dim': None,  # Input dimension (state space)
    'embedding_dim': 64,
    'n_layers': 2,
    'dropout': 0.1,
    'action_dim': None,

    'lr_encoder': 0.0003,
    'lr_decoder': 0.0003,
    'hidden_dim': 64,
    'lr':0.0003,

    # Environment settings
    'has_continuous_action_space': False,      # True: continuous action space, False: discrete
    'max_ep_len': 1000,                       # Max timesteps per episode
    'max_training_timesteps': int(3e6),       # Total training steps before stopping
    'action_std_init':0.6,

    # Logging and checkpoint frequency
    'print_freq': 1000 * 10,                  # Print avg reward every n timesteps
    'log_freq': 1000 * 2,                     # Log reward every n timesteps
    'save_model_freq': int(1e5),              # Save model every n timesteps

    # Action standard deviation settings (for continuous actions)
    'action_std': 0.6,                        # Initial std for action distribution
    'action_std_decay_rate': 0.05,            # Decay rate for action std
    'min_action_std': 0.1,                    # Minimum std after decay
    'action_std_decay_freq': int(2.5e5),      # Frequency of decay

    # PPO-specific hyperparameters
    'update_timestep': 1000 * 4,              # Update policy every n timesteps
    'K_epochs': 80,                           # PPO update epochs per batch
    'eps_clip': 0.2,                          # Clipping value for PPO surrogate loss
    'gamma': 0.99,                            # Discount factor

    # Learning rates
    #'lr_encoder': 0.0003,                     # Learning rate for encoder (actor)
    #'lr_decoder': 0.001,                      # Learning rate for decoder (critic)
    'lr_actor': 0.0003,                     # Learning rate for encoder (actor)
    'lr_critic': 0.001,                      # Learning rate for decoder (critic)

    # Misc
    'random_seed': 0                          # Random seed for reproducibility
}