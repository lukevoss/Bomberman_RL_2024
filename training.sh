# Training the Q-Learning agent with different scenarios

########## Bombs not possible ##########

# 1) Only Coins
python main.py play --agents  q_learning_agent --n-rounds 400  --scenario coin-heaven --no-gui --train 1 --save-stats

# 2) Only Crates
python main.py play --agents  q_learning_agent --n-rounds 200  --scenario crates-only --no-gui --train 1 --save-stats

# 3) Coins and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario coin-heaven --no-gui --train 1 --save-stats

# 4) Crates and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 100  --scenario crates-only --no-gui --train 1 --save-stats

# 5) Crates and Coins
python main.py play --agents  q_learning_agent --n-rounds 100  --scenario few-crates-and-coins --no-gui --train 1 --save-stats

# 6) All together (Max_Rounds = 60)
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 1000  --scenario few-crates-and-coins --no-gui --train 1 --save-stats

########## Bombs possible ##########

# 7) Only Coins
python main.py play --agents  q_learning_agent --n-rounds 400  --scenario coin-heaven --no-gui --train 1 --save-stats

# 8) Only Crates
python main.py play --agents  q_learning_agent --n-rounds 200  --scenario crates-only --no-gui --train 1 --save-stats

# 9) Only Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario empty --no-gui --train 1 --save-stats

# 10) Coins and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario coin-heaven --no-gui --train 1 --save-stats

# 11) Crates and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 400  --scenario crates-only --no-gui --train 1 --save-stats

# 12) Crates and Coins
python main.py play --agents  q_learning_agent --n-rounds 100  --scenario few-crates-and-coins --no-gui --train 1 --save-stats
