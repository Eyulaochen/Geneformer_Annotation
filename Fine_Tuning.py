import sys
import scanpy as sc
import numpy as np
from collections import Counter
from datasets import load_from_disk
from transformers.training_args import TrainingArguments
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from geneformer import DataCollatorForCellClassification

'''
# prepare the numerical labels
training = sc.read_h5ad(sys.argv[2], backed='r')
target_names = list(Counter(training.obs['celltype']))
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
np.save('2_9M_111celltype.npy', target_name_id_dict)
'''
target_name_id_dict = np.load(sys.argv[1],allow_pickle='TRUE').item()
#del training
        
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
        
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

    logging_steps = round(len(organ_trainset)/geneformer_batch_size/10)
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

    ds = load_from_disk(training_data_dir)
    organ_trainset = ds.map(classes_to_ids, num_proc=16)
    ds = load_from_disk(val_data_dir)
    organ_evalset = ds.map(classes_to_ids, num_proc=16)
    del ds

    training_args_init = TrainingArguments(**training_args)

    model = BertForSequenceClassification.from_pretrained('geneformer-12L-30M', 
                                                      num_labels=len(target_name_id_dict),
                                                      output_attentions = False,
                                                      output_hidden_states = False)
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
        )
    
    trainer.train()
    predictions = trainer.predict(organ_evalset)
    with open(f'{output_dir}predictions.pickle', 'wb') as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics('eval', predictions.metrics)
    trainer.save_model(output_dir)
    
training_data_dir = sys.argv[2]
val_data_dir = sys.argv[3]
output_dir = sys.argv[4]

FineTuning(training_data_dir, val_data_dir, output_dir)

