from typing import Optional, Dict, Any, Tuple

from numpy import where
from peft import PeftModel
from transformers import PreTrainedTokenizer

from src.domain.configuration import label_pad_token_id


def evaluate_peft_model(
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        sample: Dict[str, Any],
        max_target_length: Optional[int] = 50
) -> Tuple[str, str]:
    # generate summary
    outputs = model.generate(
        input_ids=sample["input_ids"].unsqueeze(0).cuda(),
        do_sample=True,
        top_p=0.9,
        max_new_tokens=max_target_length
    )
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace label_pad_token_id in the labels as we can't decode them.
    labels = where(sample['labels'] != label_pad_token_id, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels
