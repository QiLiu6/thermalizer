import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import thermalizer.models.misc as misc
import thermalizer.qg.performance as performance
import thermalizer.qg.util as util
import time
import os

def therm_inference_qg(identifier,start,stop,steps,forward_diff,project="therm_sweep_qg",solo_run=False):

    #wandb.init(entity="chris-pedersen",project="therm_sweep",dir="/scratch/cp3759/thermalizer_data/wandb_data")

    config={}
    config["save_dir"]="/scratch/cp3759/thermalizer_data/test_therms"
    config["identifier"]=identifier
    config["save_string"]=config["save_dir"]+"/"+config["identifier"]
    if solo_run:
        config["start"]=start
        config["stop"]=stop
    else: 
        ## This condition should fail if this is run as an individual trajectory
        config["start"]=wandb.config.start
        config["stop"]=wandb.config.stop
    config["steps"]=steps
    config["forward_diff"]=True
    config["test_suite_path"]="/scratch/cp3759/thermalizer_data/qg/test_eddy/eddy_dt5_20.pt"

    config["emulator"]="/scratch/cp3759/thermalizer_data/wandb_data/wandb/run-20241029_114648-rk44rj23/files/model_weights.pt"
    config["thermalizer"]="/scratch/cp3759/pyqg_data/wandb_runs/wandb/run-20241026_165942-p1bjxynq/files/model_weights.pt"

    model_emu=misc.load_model(config["emulator"])
    model_emu=model_emu.to("cuda")
    model_emu=model_emu.eval()

    model_therm=misc.load_diffusion_model(config["thermalizer"])
    model_therm=model_therm.to("cuda")
    model_therm=model_therm.eval()

    if solo_run:
        wandb.init(entity="chris-pedersen",project=project,dir="/scratch/cp3759/thermalizer_data/wandb_data")
        print("Saving results in directory %s" % config["save_string"])
        os.system(f'mkdir -p {config["save_string"]}')

    test_suite=torch.load(config["test_suite_path"]) ## Saved normalised?
    ## Cut test suite based on nsteps
    test_suite=test_suite[:,:config["steps"]]


    config["emulator_url"]=model_emu.config["wandb_url"]
    config["thermalizer_url"]=model_therm.config["wandb_url"]

    #wandb.init(project=project, entity="chris-pedersen",config=config,dir=config["save_string"])
    wandb.config.update(config)


    emu=performance.run_emu(test_suite[:,0],model_emu,model_therm,config["steps"],silent=True)

    ## Classify noise levels on tru sim just for continuity
    noise_classes_sim=torch.zeros(len(emu[0]),len(emu[0][1]))
    for aa in range(len(emu[0][1])):
        preds=model_therm.model.noise_class(test_suite[:,aa].to("cuda"))
        noise_classes_sim[:,aa]=preds.cpu()

    ## Run thermalizer algorithm
    start = time.time()
    algo=performance.therm_algo(test_suite[:,0],model_emu,model_therm,config["steps"],config["start"],config["stop"],config["forward_diff"],silent=False)
    end = time.time()
    algo_time=end-start
    print("Algo time =", algo_time)

    ## Ticker tape figure
    ticker=plt.figure(figsize=(20,3))
    plt.subplot(2,1,1)
    plt.title("Therm steps ticker tape; first 500")
    plt.imshow(algo[-1][:,:500])
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.title("Therm steps ticker tape; last 500")
    plt.imshow(algo[-1][:,-500:])
    plt.colorbar()
    plt.tight_layout()
    wandb.log({"Ticker tape": wandb.Image(ticker)})
    plt.close()


    ## Therm steps, full counter
    indices=[1,2,3,4,5]
    steps_full=plt.figure(figsize=(20,10))
    plt.suptitle("Thermalizing steps, full run")
    for idx in range(1,len(indices)+1):
        plt.subplot(len(indices),1,idx)
        plt.title("%d" % algo[-1][idx].sum().item())
        plt.plot(algo[-1][idx])
    wandb.log({"Therm steps full": wandb.Image(steps_full)})
    plt.close()

    ## Therm steps, zoomed
    steps_zoom=plt.figure(figsize=(20,10))
    plt.suptitle("Thermalizing steps, last 1000")
    for idx in range(1,len(indices)+1):
        plt.subplot(len(indices),1,idx)
        plt.title("%d" % algo[-1][idx][-1000:].sum().item())
        plt.plot(algo[-1][idx][-1000:])
    wandb.log({"Therm steps zoom": wandb.Image(steps_zoom)})
    plt.close()

    ## KE figure
    ke_steps=12
    ke_indices=np.linspace(1,config["steps"]-1,ke_steps,dtype=int)
    ## Now get KE and plot
    ## Get spectra to confirm
    ke_ic=util.get_ke_batch(test_suite[:,0].to("cuda"))
    ke_figure=plt.figure(figsize=(14,4.5))
    for aa in range(1,len(ke_indices)+1):
        plt.subplot(2,6,aa)
        ke_emu=util.get_ke_batch(emu[0][:,ke_indices[aa-1]].to("cuda"))
        ke_therm=util.get_ke_batch(algo[0][:,ke_indices[aa-1]].to("cuda"))
        spec_sim,nans=util.spectral_similarity(ke_ic[1],ke_therm[1])
        plt.title("Step %d, ss=%.1f" % (ke_indices[aa-1],spec_sim))
        for bb in range(len(ke_ic[1])):
            plt.loglog(ke_emu[0],ke_emu[1][bb],color="gray",alpha=0.3)
            plt.loglog(ke_therm[0],ke_therm[1][bb],color="blue",alpha=0.3)
            plt.loglog(ke_ic[0],ke_ic[1][bb],color="red",alpha=0.1)
        plt.ylim(1e-3,1e2)
        #plt.xlim(7e-1,3.5e1)
    plt.tight_layout()
    wandb.log({"Kinetic energies": wandb.Image(ke_figure)})
    plt.close()

    ## Field figures for thermalized states
    field_figure1=plt.figure(figsize=(20,12))
    preds=model_therm.model.noise_class(algo[0][:,len(algo[0][0])//2].to("cuda"))
    plt.suptitle("Thermalized states, halfway point, step %d" % (len(algo[0][0])//2))
    for aa in range(1,(5+1)):
        plt.subplot(4,5,aa)
        plt.title("%d" % preds[aa].cpu())
        plt.imshow(algo[0][aa,len(algo[0][0])//2,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+5)
        plt.imshow(algo[0][aa,len(algo[0][0])//2,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,5,aa+10)
        plt.title("%d" % preds[aa+5].cpu())
        plt.imshow(algo[0][aa+5,len(algo[0][0])//2,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+15)
        plt.imshow(algo[0][aa+5,len(algo[0][0])//2,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    wandb.log({"Fields half": wandb.Image(field_figure1)})
    plt.close()

    ## Field figures for thermalized states
    field_figure2=plt.figure(figsize=(20,12))
    preds=model_therm.model.noise_class(algo[0][:,-1].to("cuda"))
    plt.suptitle("Thermalized states, final point, step %d" % (len(algo[0][0])))
    for aa in range(1,(5+1)):
        plt.subplot(4,5,aa)
        plt.title("%d" % preds[aa].cpu())
        plt.imshow(algo[0][aa,-1,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+5)
        plt.imshow(algo[0][aa,-1,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,5,aa+10)
        plt.title("%d" % preds[aa+5].cpu())
        plt.imshow(algo[0][aa+5,-1,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+15)
        plt.imshow(algo[0][aa+5,-1,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    wandb.log({"Fields final": wandb.Image(field_figure2)})
    plt.close()

    ## Field figures for emulator
    field_figure1_emu=plt.figure(figsize=(20,12))
    preds=model_therm.model.noise_class(emu[0][:,len(emu[0][0])//2].to("cuda"))
    plt.suptitle("Emulator only, halfway point, step %d" % (len(emu[0][0])//2))
    for aa in range(1,(5+1)):
        plt.subplot(4,5,aa)
        plt.title("%d" % preds[aa].cpu())
        plt.imshow(emu[0][aa,len(emu[0][0])//2,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+5)
        plt.imshow(emu[0][aa,len(emu[0][0])//2,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,5,aa+10)
        plt.title("%d" % preds[aa+5].cpu())
        plt.imshow(emu[0][aa+5,len(emu[0][0])//2,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+15)
        plt.imshow(emu[0][aa+5,len(emu[0][0])//2,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    wandb.log({"Fields half emu": wandb.Image(field_figure1_emu)})
    plt.close()

    ## Field figures for emulator, final
    field_figure2_emu=plt.figure(figsize=(20,12))
    preds=model_therm.model.noise_class(emu[0][:,-1].to("cuda"))
    plt.suptitle("Emulator only, final point, step %d" % (len(emu[0][0])))
    for aa in range(1,(5+1)):
        plt.subplot(4,5,aa)
        plt.title("%d" % preds[aa].cpu())
        plt.imshow(emu[0][aa,-1,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+5)
        plt.imshow(emu[0][aa,-1,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,5,aa+10)
        plt.title("%d" % preds[aa+5].cpu())
        plt.imshow(emu[0][aa+5,-1,0].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,5,aa+15)
        plt.imshow(emu[0][aa+5,-1,1].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    wandb.log({"Fields final emu": wandb.Image(field_figure2_emu)})
    plt.close()

    ## Noise class figure
    fig_noise_classes=plt.figure(figsize=(18,4))
    plt.subplot(1,3,1)
    plt.title("Noise classes, first 100")
    for aa in range(len(emu[-1])):
        plt.plot(emu[-1][aa][:100],color="gray",alpha=0.3)
        plt.plot(algo[-2][aa][:100],color="blue",alpha=0.3)
        plt.plot(noise_classes_sim[aa][:100],color="red",alpha=0.1)
    plt.ylim(-2,100)
    plt.axhline(config["start"],color="black")
    plt.axhline(config["stop"],color="black")
    plt.xlabel("Emulator step")
    plt.ylabel("Predicted noise level")

    plt.subplot(1,3,2)
    plt.title("Noise classes, full run")
    for aa in range(len(emu[-1])):
        plt.plot(emu[-1][aa],color="gray",alpha=0.3)
        plt.plot(algo[-2][aa],color="blue",alpha=0.3)
        plt.plot(noise_classes_sim[aa],color="red",alpha=0.1)
    plt.ylim(-2,100)
    plt.axhline(config["start"],color="black")
    plt.axhline(config["stop"],color="black")
    plt.xlabel("Emulator step")

    plt.subplot(1,3,3)
    plt.title("Noise classes, last 100")
    for aa in range(len(emu[-1])):
        plt.plot(emu[-1][aa][-100:],color="gray",alpha=0.3)
        plt.plot(algo[-2][aa][-100:],color="blue",alpha=0.3)
        plt.plot(noise_classes_sim[aa][-100:],color="red",alpha=0.1)
    plt.ylim(-2,100)
    plt.axhline(config["start"],color="black")
    plt.axhline(config["stop"],color="black")
    plt.xlabel("Emulator step")
    wandb.log({"Noise classes": wandb.Image(fig_noise_classes)})
    plt.close()

    ## Save tensors
    if solo_run:
        for aa,em in enumerate(emu):
            torch.save(em,config["save_string"]+"/emu_%d.pt" % (aa+1))
        for aa,al in enumerate(algo):
            torch.save(al,config["save_string"]+"/therm_%d.pt" % (aa+1))
        torch.save(noise_classes_sim,config["save_string"]+"/sim_noise.pt")

    #ss_emu,nan_emu=util.spectral_similarity(ke_ic[1],ke_emu[1]) ## This bugs out due to infs/nan
    ss_therm,nan_therm=util.spectral_similarity(ke_ic[1],ke_therm[1])

    wandb.run.summary["algo time (seconds)"]=algo_time
    wandb.run.summary["spectral similarity thermalized"]=ss_therm
    wandb.run.summary["nans thermalized"]=nan_therm
    wandb.run.summary["total_therm"]=algo[-1].sum()
    torch.cuda.empty_cache()
    print("done a run")
