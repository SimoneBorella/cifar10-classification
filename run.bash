# !/bin/bash

# Training and testing
# python3 main.py --model_name CustomNet --version 0 --train --validation_split 0.2 --epochs 200 --patience 200 --lr 0.1
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file best_0.pt
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file last_0.pt

# Resume training and testing
# python3 main.py --model_name CustomNet --version 0 --train --resume --validation_split 0.2 --epochs 200 --patience 200 --lr 0.1
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file best_1.pt
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file last_1.pt

# Resume training on whole training dataset
# python3 main.py --model_name CustomNet --version 0 --train --resume --validation_split 0.0 --epochs 100 --lr 0.01 
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file last_2.pt

# In case of interruption resume from epoch
# python3 main.py --model_name CustomNet --version 0 --train --resume --resume_epoch 59 --validation_split 0.2 --epochs 200 --patience 200  --lr 0.1
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file best_1.pt
# python3 main.py --model_name CustomNet --version 0 --test --test_model_file last_1.pt
