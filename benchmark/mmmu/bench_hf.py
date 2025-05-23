import argparse
from PIL import Image
import torch
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm
from transformers import GenerationConfig


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    generation_config = GenerationConfig(
        max_new_tokens=sampling_params["max_new_tokens"],
        do_sample=False,
    )

    # ─── Model Loading ────────────────────────────────────────────────────────────
    try:
        # 1️⃣ Try the ImageText2Text interface
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        is_internvl = False
        is_causal = False

    except Exception as first_exception:
        # 2️⃣ On failure, check for InternVL or fallback to CausalLM
        try:
            if "InternVL" in args.model_path:
                # InternVL loading logic
                from internvl_utils import load_image
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                )
                model = AutoModel.from_pretrained(
                    args.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                generation_config_internvl = {
                    "max_new_tokens": sampling_params["max_new_tokens"],
                    "do_sample": False,
                }
                is_internvl = True
                is_causal = False

            else:
                # Fallback to Bunny-4B’s CausalLM interface
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                is_internvl = False
                is_causal = True

        except Exception as second_exception:
            raise RuntimeError(
                f"Failed to load model: First attempt failed with {first_exception!r}, "
                f"second attempt failed with {second_exception!r}"
            ) from second_exception

    model = model.eval().cuda()
    # ──────────────────────────────────────────────────────────────────────────────

    samples = prepare_samples(eval_args)
    out_samples = {}
    answer_dict = {}

    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        prefix, suffix = prompt.split("<")[0], prompt.split(">")[1]
        assert sample["image"] is not None

        # ─── InternVL branch ───────────────────────────────────────────────────────
        if is_internvl:
            pixel_values = load_image(sample["image_path"]).to(torch.bfloat16).cuda()
            contents = ""
            if prefix:
                contents += prefix
            contents += "<image>\n"
            if suffix:
                contents += suffix

            response = model.chat(
                tokenizer,
                pixel_values,
                contents,
                generation_config_internvl,
            )
            process_result(response, sample, answer_dict, out_samples)
            continue

        # ─── CausalLM branch ───────────────────────────────────────────────────────
        if is_causal:
            # split around the image tag
            text_before, text_after = prompt.split("<image>")
            # tokenize text before image
            input_ids = tokenizer(
                text_before, return_tensors="pt"
            ).input_ids.to(model.device)
            # process image
            image_tensor = model.process_images(
                [Image.open(sample["image_path"])],
                model.config
            ).to(dtype=model.dtype, device=model.device)
            # generate
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=sampling_params["max_new_tokens"],
                do_sample=False,
            )
            # decode only the newly generated portion
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            process_result(response, sample, answer_dict, out_samples)
            continue

        # ─── ImageText2Text branch ─────────────────────────────────────────────────
        # build the chat-style contents list
        contents = []
        if prefix:
            contents.append({"type": "text", "text": prefix})
        contents.append({"type": "image", "image": sample["image_path"]})
        if suffix:
            contents.append({"type": "text", "text": suffix})
        messages = [{"role": "user", "content": contents}]

        try:
            # template → tokens → generate
            model_inputs = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            input_len = model_inputs["input_ids"].shape[-1]
            gen = model.generate(
                **model_inputs,
                generation_config=generation_config,
            )
            gen = gen[0][input_len:]
            response = processor.decode(gen, skip_special_tokens=True)

        except Exception:
            # fallback to .chat() if templating fails
            fallback_msgs = [{"role": "user", "content": []}]
            if prefix:
                fallback_msgs[0]["content"].append(prefix)
            fallback_msgs[0]["content"].append(Image.open(sample["image_path"]))
            if suffix:
                fallback_msgs[0]["content"].append(suffix)

            response = model.chat(
                msgs=fallback_msgs,
                tokenizer=processor.tokenizer,
                sampling=False,
                max_new_tokens=sampling_params["max_new_tokens"],
                use_tts_template=False,
                generate_audio=False,
                temperature=0.0,
            )

        process_result(response, sample, answer_dict, out_samples)

    args.output_path = f"{args.model_path}_val_hf.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HF repo ID of the model weights.",
    )
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()
    eval_mmmu(args)
