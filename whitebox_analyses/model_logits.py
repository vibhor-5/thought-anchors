from model import get_token_logits_for_word, test_prompt
from pkld import pkld
import torch


@pkld
def get_logit_data(
    prompt,
    max_new_tokens=2000,
    seed=0,
    model_name="qwen-14b",
    temperature=0.6,
    float32=False,
    pos_embedding_scale=None,
    do_layers=(0, 15, 47),
    attn_layers=(0, 15, 47),
    words=("wait",),
):

    results = test_prompt(
        prompt,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        model_name=model_name,
        float32=float32,
        pos_embedding_scale=pos_embedding_scale,
        do_layers=do_layers,
        return_logits=True,
        attn_layers=attn_layers,
    )

    logits = results["logits"]
    token_texts = results["token_texts"]
    response_start = results["response_start"]
    response = results["response"]

    word2logits = {}
    for word in words:
        wait_logits = get_token_logits_for_word(logits, word, model_name=model_name)
        word2logits[word] = wait_logits

    # Clear GPU memory after processing
    torch.cuda.empty_cache()
    
    return word2logits, token_texts, response_start, response


if __name__ == "__main__":
    get_logit_data(
        prompt="",
        words=["wait"],
        model_name="qwen-15b",
    )
    # Final memory cleanup
    torch.cuda.empty_cache()
