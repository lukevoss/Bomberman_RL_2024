# Test logging over multiple runs

# Only Coins
python main.py play --agents  q_learning_agent --n-rounds 5  --scenario coin-heaven --no-gui --train 1 --save-stats

# Only Crates
python main.py play --agents  q_learning_agent --n-rounds 5  --scenario crates-only --no-gui --train 1 --save-stats

# Coins and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 5  --scenario coin-heaven --no-gui --train 1 --save-stats

# Crates and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 5  --scenario crates-only --no-gui --train 1 --save-stats

# Crates and Coins
python main.py play --agents  q_learning_agent --n-rounds 5  --scenario few-crates-and-coins --no-gui --train 1 --save-stats

# All together (Max_Rounds = 60)
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 10  --scenario few-crates-and-coins --no-gui --train 1 --save-stats
