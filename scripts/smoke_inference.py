from pathlib import Path
import torch
from src.model import build_model, mc_predict, mc_predict_multilabel
from src.gradcam import GradCAM
from src.ensemble import ensemble_predict

def load(path):
    ckpt=torch.load(path,map_location="cpu")
    model=build_model(len(ckpt["classes"]),dropout=float(ckpt.get("dropout",0.30)),pretrained=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model,ckpt

def main():
    path=Path("artifacts/cxr_mobilenetv3.pt")
    assert path.exists(), "checkpoint missing"
    model,ckpt=load(path)
    size=int(ckpt.get("input_size",160))
    x=torch.randn(1,3,size,size)

    mean,var,entropy,mi=mc_predict(model,x,passes=3)
    assert mean.shape==(1,2) and torch.isfinite(mean).all()
    assert torch.isfinite(var).all() and torch.isfinite(entropy).all() and torch.isfinite(mi).all()

    engine=GradCAM(model,model.features[-1])
    cam,idx=engine(x); engine.close()
    assert cam.ndim==2 and cam.min()>=0 and cam.max()<=1.000001

    ensemble_paths=[Path("artifacts/cxr_mobilenetv3.pt"),Path("artifacts/cxr_seed_2027.pt"),Path("artifacts/cxr_seed_2028.pt")]
    if all(p.exists() for p in ensemble_paths):
        models=[load(p)[0] for p in ensemble_paths]
        emean,evar,eent,emi=ensemble_predict(models,x)
        assert emean.shape==(1,2) and torch.isfinite(emean).all()
        assert torch.isfinite(evar).all() and torch.isfinite(eent).all() and torch.isfinite(emi).all()

    nih=Path("artifacts/cxr_nih_multilabel.pt")
    if nih.exists():
        nmodel,nckpt=load(nih)
        nx=torch.randn(1,3,int(nckpt.get("input_size",128)),int(nckpt.get("input_size",128)))
        nmean,nvar,nent,nmi=mc_predict_multilabel(nmodel,nx,passes=3)
        assert nmean.shape==(1,len(nckpt["classes"]))
        assert torch.isfinite(nmean).all() and torch.isfinite(nvar).all()
        assert torch.isfinite(nent).all() and torch.isfinite(nmi).all()

    print({
        "binary_classes":ckpt["classes"],
        "input_size":size,
        "pred_index":idx,
        "probabilities":mean.tolist(),
        "entropy":entropy.tolist(),
        "mi":mi.tolist(),
        "cam_shape":cam.shape,
        "ensemble_present":all(p.exists() for p in ensemble_paths),
        "nih_present":nih.exists(),
    })

if __name__=="__main__":
    main()
