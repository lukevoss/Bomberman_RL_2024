cd C:\Users\luke\OneDrive\Dokumente\UniHeidelberg\Master\Semester2\MLE\Final\Bomberman_RL
conda activate ml_homework
python main.py play --my-agent ppo_agent --n-rounds 1000 --train 1 --no-gui --scenario coin-heaven

python main.py play --agents ppo_agent2 --train 1 --n-rounds 30 --no-gui --scenario coin-heaven
