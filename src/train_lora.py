from __future__ import annotations
import argparse, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg.get('model_name', 'gpt2')
    out_dir = cfg.get('output_dir', 'models/lora-gpt2')
    train_file = cfg['train_file']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset('json', data_files=train_file, split='train')
    eval_ratio = float(cfg.get('eval_ratio', 0.02))
    n_eval = max(1, int(len(ds)*eval_ratio))
    ds_train = ds.select(range(len(ds)-n_eval))
    ds_eval  = ds.select(range(len(ds)-n_eval, len(ds)))

    def to_text(ex):
        ex['text'] = ex['prompt'] + ex['completion']
        return ex
    ds_train = ds_train.map(to_text)
    ds_eval  = ds_eval.map(to_text)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    lcfg = LoraConfig(r=cfg['lora']['r'], lora_alpha=cfg['lora']['alpha'], lora_dropout=cfg['lora']['dropout'], bias='none', target_modules=['c_attn','q_proj','v_proj'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)

    args_tr = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],
        num_train_epochs=cfg['num_train_epochs'],
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    def tok(ex): return tokenizer(ex['text'], truncation=True, max_length=cfg['block_size'])
    ds_train_tok = ds_train.map(tok, batched=True, remove_columns=ds_train.column_names)
    ds_eval_tok  = ds_eval.map(tok, batched=True, remove_columns=ds_eval.column_names)

    trainer = Trainer(model=model, args=args_tr, train_dataset=ds_train_tok, eval_dataset=ds_eval_tok, data_collator=collator)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    main()
