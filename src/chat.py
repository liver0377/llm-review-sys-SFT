from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc


def chat(model_path: str = "outputs/paper_review_qwen2.5_7b_lora"):
    args = {
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_name_or_path": model_path,
        "template": "qwen",
        "finetuning_type": "lora",
    }

    chat_model = ChatModel(args)
    messages = []

    print("\n" + "=" * 60)
    print("论文评审模型 - 对话测试")
    print("输入论文内容进行评审，输入 'exit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            response = ""
            print("Assistant: ", end="", flush=True)
            for new_text in chat_model.stream_chat(messages):
                print(new_text, end="", flush=True)
                response += new_text
            print("\n")
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n已中断")
            break

    torch_gc()
    print("\n再见！")


if __name__ == "__main__":
    import sys

    model_path = (
        sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_review_qwen2.5_7b_lora"
    )
    chat(model_path)
