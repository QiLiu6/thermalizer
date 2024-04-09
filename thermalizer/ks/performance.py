import matplotlib.pyplot as plt
import torch

def plot_rollout_state(valid_data,model,index):
    ## State Rollout
    model=model.eval()
    x_now=torch.tensor(valid_data[index][0],requires_grad=False).unsqueeze(0).unsqueeze(0)

    x_pred=torch.empty(1000,128)
    x_pred[0]=x_now.squeeze()
    for aa in range(1,1000):
        x_now=model(x_now)
        x_pred[aa]=x_now.squeeze()

    plt.figure(figsize=(12,6))
    plt.subplot(3,1,1)
    plt.title("True")
    plt.imshow(valid_data[index].T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.title("Emulator")
    plt.imshow(x_pred.detach().numpy().T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.title("Residual")
    plt.imshow(x_pred.detach().T-valid_data[index].T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()


def plot_rollout_residual(valid_data,model,index):
    ## Residual Rollout
    model=model.eval()
    x_now=torch.tensor(valid_data[index][0],requires_grad=False).unsqueeze(0).unsqueeze(0)
    
    x_pred=torch.empty(1000,128)
    x_pred[0]=x_now.squeeze()
    for aa in range(1,1000):
        x_now=model(x_now)+x_now
        x_pred[aa]=x_now.squeeze()

    plt.figure(figsize=(12,6))
    plt.subplot(3,1,1)
    plt.title("True")
    plt.imshow(valid_data[index].T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()
    
    plt.subplot(3,1,2)
    plt.title("Emulator")
    plt.imshow(x_pred.detach().numpy().T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()
    
    plt.subplot(3,1,3)
    plt.title("Residual")
    plt.imshow(x_pred.detach().T-valid_data[index].T,cmap="inferno")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.colorbar()
