import scanpy as sc
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
evalset = ds.map(classes_to_ids, num_proc=16)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }

def FineTuning(training_data_dir, val_data_dir, output_dir):
    # set model parameters
    # max input size
    max_input_size = 2**11  # 2048
    # set training hyperparameters
    # max learning rate
    max_lr = 5e-5
    # how many pretrained layers to freeze
    freeze_layers = 0
    # number gpus
    num_gpus = 1
    # number cpu cores
    num_proc = 16
    # batch size for training and eval
    geneformer_batch_size = 12
    # learning schedule
    lr_schedule_fn = 'linear'
    # warmup steps
    warmup_steps = 500
    # number of epochs
    epochs = 10
    # optimizer
    optimizer = 'adamw'

    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
        }
    
    
    
