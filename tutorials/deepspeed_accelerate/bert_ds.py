import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification,pipeline
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
from evaluate import evaluator
from datasets import load_dataset


# Model Repository on huggingface.co
model_id = "dslim/bert-large-NER"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.half, # dtype of the weights (fp16)
    # injection_policy={"BertLayer" : HFBertLayerPolicy}, # replace BertLayer with DS HFBertLayerPolicy
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)

# create acclerated pipeline
ds_clf = pipeline("token-classification", model=ds_model, tokenizer=tokenizer,device=0)

# Test pipeline
# example = "My name is Wolfgang and I live in Berlin"
# ner_results = ds_clf(example)
# print(ner_results)

# define evaluator
task_evaluator = evaluator("token-classification")

# run baseline
results = task_evaluator.compute(
    model_or_pipeline=token_clf,
    data=eval_dataset,
    metric="seqeval",
)

# run baseline
ds_results = task_evaluator.compute(
    model_or_pipeline=ds_clf,
    data=eval_dataset,
    metric="seqeval",
)

print(f"Overall f1 score for our model is {ds_results['overall_f1']*100:.2f}%")
print(f"The avg. Latency of the model is {ds_results['latency_in_seconds']*1000:.2f}ms")