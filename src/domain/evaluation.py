from typing import Optional, Dict, Any, Tuple
from time import time

from numpy import where
from peft import PeftModel
from transformers import PreTrainedTokenizer
from streamlit import write

from src.domain.configuration import label_pad_token_id


def evaluate_peft_model(
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        sample: Dict[str, Any],
        max_target_length: Optional[int] = 50
) -> Tuple[str, str]:
    # generate summary
    write(sample)
    t1 = time()
    sample_cuda = sample["input_ids"].unsqueeze(0).cuda()
    t2 = time()
    write(f"Time to move sample to GPU: {t2 - t1:.2f} sec.")
    write(sample_cuda)
    """
    outputs = model.generate(
        input_ids=sample["input_ids"].unsqueeze(0).cuda(),
        do_sample=True,
        top_p=0.9,
        max_new_tokens=max_target_length
    )
    """
    t3 = time()
    outputs = model.generate(
        input_ids=sample_cuda,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=max_target_length
    )
    t4 = time()
    write(f"Time to generate output: {t4 - t3:.2f} sec.")
    write(outputs)

    t5 = time()
    outputs_numpy = outputs[0].detach().cpu().numpy()
    t6 = time()
    write(f"Time to move output to numpy: {t6 - t5:.2f} sec.")
    write(outputs_numpy)
    # prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    t7 = time()
    prediction = tokenizer.decode(outputs_numpy, skip_special_tokens=True)
    t8 = time()
    write(f"Time to make prediction: {t8 - t7:.2f} sec.")
    write(prediction)
    # decode eval sample
    # Replace label_pad_token_id in the labels as we can't decode them.
    t9 = time()
    labels = where(sample['labels'] != label_pad_token_id, sample['labels'], tokenizer.pad_token_id)
    t10 = time()
    write(f"Time to Replace label_pad_token_id: {t10 - t9:.2f} sec.")
    t11 = time()
    labels = tokenizer.decode(labels, skip_special_tokens=True)
    t12 = time()
    write(f"Time to decode label {t12 - t11:.2f} sec.")
    write(labels)

    # Some simple post-processing
    return prediction, labels
