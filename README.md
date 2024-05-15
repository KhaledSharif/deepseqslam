# DeepSeqSLAM
## A Trainable NN for Joint Global Description and Sequential Place Recognition

<img src="https://user-images.githubusercontent.com/25828032/109748988-87dd6600-7c25-11eb-82c4-b8c3d298601a.png" />

DeepSeqSLAM is a CNN + LSTM baseline architecture for state-of-the-art route-based place recognition. It leverages visual and positional time-series data for joint global description and sequential place inference in the context of simultaneous localization and mapping (SLAM) and autonomous driving research. Contrary to classical two-stage pipelines, *e.g.*, *match-then-temporally-filter*, this codebase is orders of magnitud faster, scalable and learns from a single traversal of a route,
while accurately generalizing to multiple traversals of the same route under very different environmental conditions.

---

The challenging Gardens Point Walking dataset consists of three folders with 200 images each. The image name indicates correspondence in location between each of the three route traversals. Download the [dataset](https://zenodo.org/record/4745641), unzip, and place the `day_left`, `day_right`, and `night_right` image folders in the `datasets/GardensPointWalking` directory of DeepSeqSLAM.

```bash
SEQ_LENGTH=10
BATCH_SIZE=16
EPOCHS=100
NGPUS=1

SEQ1='day_left'
SEQ2='day_right'
SEQ3='night_right'

# alexnet, resnet18, vgg16, squeezenet1_0, densenet161
CNN='resnet18'

MODEL_NAME="gp_${CNN}_lstm"

python run.py train \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGTH \
    --epochs $EPOCHS \
    --val_set $SEQ2 \
    --cnn_arch $CNN

for i in $SEQ1 $SEQ2 $SEQ3
do
    python run.py val \
    --model_name $MODEL_NAME \
    --ngpus $NGPUS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LENGTH \
    --val_set $i \
    --cnn_arch $CNN
done
```

```bash
usage: run.py [-h] [--data_path DATA_PATH] [-o OUTPUT_PATH]
              [--model_name MODEL_NAME] [-a ARCH] [--pretrained PRETRAINED]
              [--val_set VAL_SET] [--ngpus NGPUS] [-j WORKERS]
              [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
              [--load LOAD] [--nimgs NIMGS] [--seq_len SEQ_LEN]
              [--nclasses NCLASSES] [--img_size IMG_SIZE]

  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to dataset folder that contains preprocessed
                        train and val *npy image files
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path for storing model checkpoints
  --model_name MODEL_NAME
                        checkpoint model name (default:
                        deepseqslam_resnet18_lstm)
  -a ARCH, --cnn_arch ARCH
                        model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn (default: resnet18)
  --pretrained PRETRAINED
                        use pre-trained CNN model (default: True)
  --val_set VAL_SET     validation_set (default: day_right)
  --ngpus NGPUS         number of GPUs for training; 0 if you want to run on
                        CPU (default: 2)
  -j WORKERS, --workers WORKERS
                        number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 200)
  --batch_size BATCH_SIZE
                        mini-batch size: 2^n (default: 32)
  --lr LR, --learning_rate LR
                        initial learning rate (default: 1e-3)
  --load LOAD           restart training from last checkpoint
  --nimgs NIMGS         number of images (default: 200)
  --seq_len SEQ_LEN     sequence length: ds (default: 10)
  --nclasses NCLASSES   number of classes = nimgs - seq_len (default: 190)
  --img_size IMG_SIZE   image size (default: 224)
```

