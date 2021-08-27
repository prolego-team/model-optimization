"""
adventure game for figuring out how to optimize your model
"""

while True:
    print("So you want to reduce the size of your trained transformer model?")
    response = input().lower()
    if response == "quit":
        break
    if response == "no":
        print("Well that makes things easy - just use HuggingFace Trainer for inference.")
        print()
        break
    if response == "yes":
        print("Are you willing to retrain the model?")
        response = input().lower()
        if response == "yes":
            print("Try DistilRoBERTa")
            print()
        else:
            print("Do you need to use AWS Elastic Inference?")
            serialization_response = input().lower()
            print("Is it OK if model performance changes somewhat?")
            quantization_response = input().lower()
            print()
            if serialization_response == "yes":
                print("Serialize the model with TorchScript.")
            else:
                print("Serialize the model with ONNX.")
            if quantization_response == "no":
                print("You're done!")
                print()
                break
            else:
                quantization_method = "torch.quantize.dynamic_quantization" if serialization_response == "yes" else "ORT"
                print("Quantize the model with " + quantization_method)
            print()
            print("Do you want to do some more work to make the model a little smaller?")
            response = input().lower()
            if response == "no":
                print()
                print("You're done!")
                print()
                break
            else:
                print()
                print("Prune some of the attention heads, then perform serialization and quantization.")
                print("You're done!")
                print()
                break
