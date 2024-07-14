The models of examples of Pytorch_Dart will be stored here.
Also,you can generate the model by yourself.
Run code below to generate traced_resnet_model.pt
'''python

    import torch
    import torchvision

    # An instance of your model.
    model = torchvision.models.resnet18(pretrained=True)

    # Switch the model to eval model
    model.eval()

    # An example input you would normally provide to your model's forward() method.
        example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save the TorchScript model
    traced_script_module.save("traced_resnet_model.pt")

