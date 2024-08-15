cd C:\Users\luke\OneDrive\Dokumente\UniHeidelberg\Master\Semester2\MLE\Final\Bomberman_RL
conda activate ml_homework
:::::::::::::::::::: No Bombs possible: ::::::::::::::::::::
:: settings.Max_Rounds = 100, 
:: gamma = 0, 
:: LR = 0.9, 
:: max_epsilon = 0.8, 
:: min_epsilon = 0.05, 
:: decay_rate = 0.001

:: Only Coins
python main.py play --agents  q_learning_agent --n-rounds 400  --scenario coin-heaven --no-gui --train 1

:: Only Crates
python main.py play --agents  q_learning_agent --n-rounds 200  --scenario crates-only --no-gui --train 1

:: Coins and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario coin-heaven --no-gui --train 1

:: Crates and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 100  --scenario crates-only --no-gui --train 1

:: Crates and Coins
python main.py play --agents  q_learning_agent --n-rounds 100  --scenario few-crates-and-coins --no-gui --train 1

:: All together (Max_Rounds = 60)
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 1000  --scenario few-crates-and-coins --no-gui --train 1


:::::::::::::::::::: Bombs possible: ::::::::::::::::::::
:: settings.Max_Rounds = 100, 
:: gamma = 0, 
:: LR = 0.9, 
:: max_epsilon = 0.8, 
:: min_epsilon = 0.05, 
:: decay_rate = 0.001

:: Only Coins
python main.py play --agents  q_learning_agent --n-rounds 400  --scenario coin-heaven --no-gui --train 1

:: Only Crates
python main.py play --agents  q_learning_agent --n-rounds 200  --scenario crates-only --no-gui --train 1

:: Only Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario empty --no-gui --train 1

:: Coins and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 200  --scenario coin-heaven --no-gui --train 1

:: Crates and Opponents
python main.py play --agents  q_learning_agent rule_based_agent --n-rounds 400  --scenario crates-only --no-gui --train 1

:: Crates and Coins
python main.py play --agents  q_learning_agent --n-rounds 100  --scenario few-crates-and-coins --no-gui --train 1