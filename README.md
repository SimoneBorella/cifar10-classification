# cifar10-classification
Classification on cifar10 dataset with several models


## Results

Model: `CustomNet`

Optimizer: `SGD(lr, momentum=0.9, weight_decay=0.0005)`

Scheduler: `CosineAnnealingLR(lr)`

| MODEL VERSION | LR    | NUM_EPOCHS | TEST ACCURACY |
|---------------|-------|------------|-----------|
| 0             | 0.01  | 100        | 0.704     |
| 1             | 0.1   | 100        | 0.753     |
| **2**             | **0.1**   | **200**        | **0.763**     |



Model: `DLA`

Optimizer: `SGD(lr, momentum=0.9, weight_decay=0.0005)`

Scheduler: `CosineAnnealingLR(lr)`

| MODEL VERSION | LR    | NUM_EPOCHS | TEST ACCURACY |
|---------------|-------|------------|-----------|
| **0**             | **0.1**   | **200**        | **0.949**     |


Model: `EfficientNetB0`

Optimizer: `SGD(lr, momentum=0.9, weight_decay=0.0005)`

Scheduler: `CosineAnnealingLR(lr)`

| MODEL VERSION | LR    | NUM_EPOCHS | TEST ACCURACY |
|---------------|-------|------------|-----------|
| 0             | 0.01  | 100        | 0.847     |
| **1**             | **0.1**   | **200**        | **0.891**     |


Model: `MobileNetV2`

Optimizer: `SGD(lr, momentum=0.9, weight_decay=0.0005)`

Scheduler: `CosineAnnealingLR(lr)`

| MODEL VERSION | LR    | NUM_EPOCHS | TEST ACCURACY |
|---------------|-------|------------|-----------|
| **0**             | **0.1**   | **200**        | **0.917**     |



Model: `ResNet18`

Optimizer: `SGD(lr, momentum=0.9, weight_decay=0.0005)`

Scheduler: `CosineAnnealingLR(lr)`

| MODEL VERSION | LR    | NUM_EPOCHS | TEST ACCURACY |
|---------------|-------|------------|-----------|
| 0             | 0.01  | 100        | 0.877     |
| **1**             | **0.1**   | **200**        | **0.950**     |