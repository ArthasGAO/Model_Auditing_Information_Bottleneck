"""Evaluate the newer CIFAR-10 FGSM positives in one separate master table.

Default: final numbered checkpoint of each of nine evasion runs, plus three
pre-evasion controls. Uses existing baseline implementations and victim caches.
Run --preflight first. --all-epochs expands the evasion runs to every epoch.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
METHODS = ('IPGuard', 'DeepJudge', 'ADV_TRA', 'DI')
IDENTITY = ['Case_ID', 'Scenario_Name', 'Stage', 'FT_Method', 'Evasion_Attack',
            'Evasion_Epsilon', 'Epoch', 'Checkpoint_Path', 'Checkpoint_SHA256',
            'Victim_Arch', 'Suspect_Arch', 'Dataset', 'Suspect_Type', 'Config_ID']
METRICS = {
    'IPGuard': ['TR', 'TL', 'RR', 'RL', 'Mean_Matching_Rate'],
    'DeepJudge': ['Rob_Victim', 'Rob_Suspect', 'RobD', 'JSD', 'Tau_RobD',
                  'Tau_JSD', 'RobD_Vote', 'JSD_Vote', 'P_Copy', 'Stolen'],
    'ADV_TRA': ['Detection_Rate', 'Mean_Mutation_Rate', 'Num_Trajectories', 'Stolen'],
    'DI': ['Mean_Private', 'Mean_Public', 'Delta', 'T_Stat', 'P_Value', 'Stolen'],
}
COLUMNS = IDENTITY + [f'{m}_{k}' for m in METHODS
                      for k in ['Status', 'Updated_At', 'Seconds', 'Error', *METRICS[m]]]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def numbered_checkpoints(directory):
    return sorted(directory.glob('epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))


def discover_cases(all_epochs=False):
    cases = []
    # Show the FT-AL control and strongest evasion first within each phase.
    for ft in ('FT-AL', 'FT-LL', 'RT-AL'):
        base = f'CIFAR-10_ResNet-18_25000_Same_10000_42_1.0_{ft}_ftsize=10000_ftseed=0'
        dirs = [('pre_evasion', '', ROOT/'saved_models/ft_vanilla'/base)]
        dirs += [('post_evasion', eps, ROOT/'saved_models/at_train_new1'/
                  f'{base}_FGSM_eps={eps}_atseed=42')
                 for eps in ('0.031373', '0.015686', '0.007843')]
        for stage, eps, directory in dirs:
            checkpoints = numbered_checkpoints(directory)
            if not checkpoints:
                raise FileNotFoundError(f'No numbered checkpoints: {directory}')
            selected = checkpoints if all_epochs and stage == 'post_evasion' else checkpoints[-1:]
            for path in selected:
                epoch = int(path.stem.split('_')[-1])
                cases.append(dict(Case_ID=f'{directory.name}_epoch={epoch}',
                    Scenario_Name=directory.name, Stage=stage, FT_Method=ft,
                    Evasion_Attack='FGSM' if eps else 'none', Evasion_Epsilon=eps,
                    Epoch=epoch, Checkpoint_Path=path.as_posix(),
                    Checkpoint_SHA256=sha256(path), Victim_Arch='ResNet-18',
                    Suspect_Arch='ResNet-18', Dataset='CIFAR-10', Suspect_Type='positive'))
    return cases


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def write_master(path, rows):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file
    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)
    def flush(self):
        self.original.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'saved_logs/at_eval/combined_ft_evasion_2026-09-09')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--all-epochs', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from util import build_dataset_from_yaml
    from util_adv import load_victim_model, load_positive_suspect
    from main_DI_eval import set_seed
    from main_DEEPJUDGE_eval import compute_robustness, compute_robd, compute_jsd
    from AdvAttack.IP_Guard import verify_fingerprint
    from AdvAttack.DI import DatasetInferencePipeline
    from AdvAttack.advtra.adv_tra_adapter import build_args, run_verification_pretty

    if not torch.cuda.is_available():
        raise RuntimeError('This evaluation requires the existing CUDA environment.')
    torch.set_num_threads(4)
    device = 'cuda'
    set_seed(42)
    cases = discover_cases(args.all_epochs)
    victim_id = 'CIFAR-10_25000_seed=42_overlap=1.0'
    victim_path = ROOT/'saved_models/vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0/best_epoch.pth'
    ip_dir = ROOT/f'Indices/CIFAR-10/IPGuard/victim=ResNet-18_{victim_id}_k=5_size=100'
    adv_path = ROOT/'Indices/adv_examples/CIFAR-10_ResNet-18_PGD_eps=0.03_step_size=0.003_steps=10.pt'
    dj_dir = ROOT/'saved_logs/at_eval/DeepJudge/Threshold'
    tau_paths = {metric: dj_dir/metric/f'victim=ResNet-18_{victim_id}_{metric}_{suffix}_threshold.json'
                 for metric, suffix in [('RobD','PGD_eps=0.03_step_size=0.003_steps=10'),('JSD','clean')]}
    trajectory_root = ROOT/'results/advtra1/CIFAR-10_ResNet-18_25000_42_1.0/fingerprints'
    trajectory_dir = trajectory_root/'cifar10/trajectory_8'
    index_path = ROOT/'Indices/CIFAR-10/group_A_subset_10000_from_25000_seed42.npy'
    assets = [victim_path, adv_path, index_path, *tau_paths.values(),
              *(ip_dir/f'{tag}.pt' for tag in ('TR','TL','RR','RL'))]
    assets += [trajectory_dir/str(i)/name for i in range(1,101) for name in ('tra_log.pth','pred_log.pth')]
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required existing cache: {path}')
    idx = np.load(index_path)
    if len(idx) < 2000 or len(set(idx[:2000].tolist())) != 2000:
        raise ValueError('DI requires 2000 distinct private indices for separate train/verify sets.')
    config = dict(protocol='four_baselines_ft_evasion_v1', seed=42,
        victim_path=victim_path.as_posix(), victim_sha256=sha256(victim_path),
        checkpoint_selection='all evasion epochs' if args.all_epochs else 'highest numbered epoch',
        ipguard=dict(k=5,size=100,configs=['TR','TL','RR','RL'],cache=ip_dir.as_posix(),decision='score only; no calibrated threshold'),
        deepjudge=dict(attack='PGD',eps=0.03,steps=10,step_size=0.003,n_test=10000,
                       thresholds={k:p.as_posix() for k,p in tau_paths.items()}),
        adv_tra=dict(num_trajectories=100,length=8,threshold=0.5,cache=trajectory_root.as_posix()),
        di=dict(n_train_per_set=1000,n_verify_per_set=1000,alpha=0.05,epochs=30,
                n_samples=10,max_steps=50,noise_uniform=0.005,noise_gaussian=0.005,
                noise_laplace=0.01,point_batch_size=8,preprocessing='clean raw [0,1]',
                private_train_indices=idx[:1000].tolist(),private_verify_indices=idx[1000:2000].tolist(),
                public_train_indices=list(range(1000)),public_verify_indices=list(range(1000,2000))),
        asset_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in assets},
        implementation_sha256={p:sha256(ROOT/p) for p in [Path(__file__).name,'AdvAttack/DI.py',
            'AdvAttack/IP_Guard.py','main_DEEPJUDGE_eval.py','util_adv.py',
            'AdvAttack/advtra/adv_tra_adapter.py','AdvAttack/advtra/advtra_vendored/adv_gen.py']},
        torch_version=torch.__version__)
    config_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    for c in cases:
        c['Config_ID'] = config_id
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', cases=len(cases), config_id=config_id,
                             checkpoints=[c['Checkpoint_Path'] for c in cases],
                             baseline_assets=len(assets), output=str(args.output)),indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding='utf-8'))
        if old['config_id'] != config_id or old['cases'] != cases:
            raise RuntimeError('Existing output has different inputs/settings. Choose a new --output directory.')
    else:
        atomic_json(manifest,dict(config_id=config_id,config=config,cases=cases))
    master = args.output/'master.csv'
    if master.exists():
        with master.open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        if [r['Case_ID'] for r in rows] != [c['Case_ID'] for c in cases]:
            raise RuntimeError('Master case identities do not match the manifest.')
    else:
        rows = [{**dict.fromkeys(COLUMNS,''), **c,
                 **{f'{m}_Status':'pending' for m in METHODS}} for c in cases]
        write_master(master, rows)
    log = (args.output/'run.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout,log), Tee(sys.stderr,log)
    status_path=args.output/'status.json'
    def status(state, **extra):
        atomic_json(status_path,dict(state=state,pid=os.getpid(),updated_at=datetime.now().isoformat(),
            completed=sum(r[f'{m}_Status']=='complete' for r in rows for m in METHODS),
            failed=sum(r[f'{m}_Status']=='failed' for r in rows for m in METHODS),
            total=len(cases)*len(METHODS),**extra))
    status('starting')
    print(f'Run {config_id}: {len(cases)} positive checkpoints; master={master}',flush=True)
    ds_cfg=dict(name='CIFAR-10',normalization='cifar10',img_size=32,group_size=25000,download=False)
    ds,num_classes,_=build_dataset_from_yaml(ds_cfg)
    victim=load_victim_model(dict(Model='ResNet-18',Model_Name='CIFAR-10_ResNet-18_25000',Dataset=ds_cfg),ds,num_classes,42)
    raw_loader=DataLoader(ds.raw_test_set,batch_size=128,shuffle=False,num_workers=0)
    def load_case(row):
        return load_positive_suspect('ResNet-18',num_classes,ds,Path(row['Checkpoint_Path']))
    evaluators={}

    for method in METHODS:
        pending=[r for r in rows if r[f'{method}_Status']!='complete']
        if not pending:
            continue
        print(f'\n=== Preparing {method} ===',flush=True)
        status('preparing',baseline=method)
        try:
            set_seed(42)
            if method=='IPGuard':
                fps={tag:torch.load(ip_dir/f'{tag}.pt',map_location='cpu',weights_only=False) for tag in ('TR','TL','RR','RL')}
                def evaluate(model):
                    values={tag:verify_fingerprint(model,fp,device=device,batch_size=100)['matching_rate'] for tag,fp in fps.items()}
                    values['Mean_Matching_Rate']=float(np.mean(list(values.values())))
                    return values
            elif method=='DeepJudge':
                adv=torch.load(adv_path,map_location='cpu',weights_only=False)
                taus={k:json.loads(p.read_text())['tau'] for k,p in tau_paths.items()}
                rob_v=compute_robustness(victim,adv,device=device,batch_size=128)
                def evaluate(model):
                    rob_s=compute_robustness(model,adv,device=device,batch_size=128)
                    robd=compute_robd(rob_v,rob_s)
                    jsd=compute_jsd(victim,model,raw_loader,device=device)
                    vr,vj=int(robd<=taus['RobD']),int(jsd<=taus['JSD'])
                    return dict(Rob_Victim=rob_v,Rob_Suspect=rob_s,RobD=robd,JSD=jsd,
                        Tau_RobD=taus['RobD'],Tau_JSD=taus['JSD'],RobD_Vote=vr,JSD_Vote=vj,
                        P_Copy=(vr+vj)/2,Stolen=int(vr+vj==2))
            elif method=='ADV_TRA':
                tra_args=build_args(dataset_name='cifar10',num_classes=10,
                    data_path=str(args.output/'advtra_data'),model_path=str(args.output/'advtra_dummy'),
                    fingerprint_path=str(trajectory_root),num_trajectories=100,length=8,
                    tra_classes=10,threshold=0.5,device=device)
                sanity=run_verification_pretty(tra_args,victim)
                if sanity.detection_rate<0.99 or sanity.num_trajectories!=100:
                    raise RuntimeError(f'ADV_TRA victim-cache sanity failed: {sanity}')
                def evaluate(model):
                    result=run_verification_pretty(tra_args,model)
                    return dict(Detection_Rate=result.detection_rate,
                        Mean_Mutation_Rate=result.mean_mutation_rate,Num_Trajectories=result.num_trajectories,
                        Stolen=int(result.detection_rate>0.5))
            else:
                pipeline=DatasetInferencePipeline(victim,point_batch_size=8,device=device)
                regressor_path=args.output/'di_regressor_clean_disjoint.pt'
                if regressor_path.exists():
                    pipeline.load_regressor(regressor_path)
                else:
                    pipeline.train_regressor(ds.raw_train_clean_set,ds.raw_test_set,
                        private_indices=idx[:1000],public_indices=list(range(1000)),
                        n_train_samples=1000,regressor_epochs=30)
                    pipeline.save_regressor(regressor_path)
                def evaluate(model):
                    result=pipeline.verify_suspect(model,ds.raw_train_clean_set,ds.raw_test_set,
                        private_indices=idx[1000:2000],public_indices=list(range(1000,2000)),
                        n_test_samples=1000,alpha=0.05)
                    return {name: int(result[key]) if key=='stolen' else result[key] for name,key in
                        [('Mean_Private','mean_private'),('Mean_Public','mean_public'),('Delta','delta'),
                         ('T_Stat','t_stat'),('P_Value','p_value'),('Stolen','stolen')]}
        except Exception:
            error=traceback.format_exc()
            print(error,flush=True)
            for row in pending:
                row[f'{method}_Status']='failed'
                row[f'{method}_Error']=error
            write_master(master,rows)
            status('running',baseline=method,error=error)
            continue
        for row in pending:
            start=time.monotonic()
            model=None
            row[f'{method}_Status']='running'
            write_master(master,rows)
            status('running',baseline=method,case_id=row['Case_ID'])
            print(f"\n[{method}] {row['Case_ID']}",flush=True)
            try:
                set_seed(42)
                model=load_case(row)
                result=evaluate(model)
                if not all(np.isfinite(v) for v in result.values()):
                    raise ValueError(f'Non-finite evaluation metric: {result}')
                row.update({f'{method}_{k}':v for k,v in result.items()})
                row[f'{method}_Status']='complete'
                row[f'{method}_Error']=''
                print(f'RESULT {method}: {json.dumps(result)}',flush=True)
            except Exception:
                row[f'{method}_Status']='failed'
                row[f'{method}_Error']=traceback.format_exc()
                print(row[f'{method}_Error'],flush=True)
            finally:
                row[f'{method}_Seconds']=round(time.monotonic()-start,3)
                row[f'{method}_Updated_At']=datetime.now().isoformat(timespec='seconds')
                write_master(master,rows)
                del model
                torch.cuda.empty_cache()
                status('running',baseline=method,case_id=row['Case_ID'])
    failed=sum(r[f'{m}_Status']!='complete' for r in rows for m in METHODS)
    status('completed_with_errors' if failed else 'complete')
    print(f'Finished: {len(cases)*4-failed}/{len(cases)*4} evaluations complete. {master}',flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
