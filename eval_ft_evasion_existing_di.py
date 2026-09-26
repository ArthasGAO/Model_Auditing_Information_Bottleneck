"""Reuse the original victim regressor for the existing combined positive sweep.

No regressor training is performed. Other baseline columns are preserved.
The initial replacement archives the earlier master and its status/manifest.
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
import run_four_baselines_ft_evasion as common
from launch_four_baselines_ft_evasion import retry_file_write

ROOT = common.ROOT
OUTPUT = ROOT/'saved_logs/at_eval/combined_ft_evasion_2026-09-09'
OLD_REGRESSOR = ROOT/'saved_models/di_regressor/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0_n=1000.pt.augmented_stale.bak'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    import numpy as np
    import torch
    from AdvAttack.DI import DatasetInferencePipeline, ConfidenceRegressor
    from util import build_dataset_from_yaml
    from util_adv import load_positive_suspect
    from main_DI_eval import set_seed
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('The CUDA environment is required.')
    checkpoint=torch.load(OLD_REGRESSOR,map_location='cpu',weights_only=True)
    # Explicitly restore the feature-generation parameters stored with the old model.
    walk_keys=['n_samples','noise_uniform','noise_gaussian','noise_laplace','max_steps']
    walk={key:checkpoint[key] for key in walk_keys}
    if checkpoint['embedding_dim'] != 3*walk['n_samples']:
        raise ValueError('Checkpoint embedding dimensions do not match its walk settings.')
    regressor=ConfidenceRegressor(input_dim=checkpoint['embedding_dim'],hidden_dim=checkpoint['embedding_dim'])
    regressor.load_state_dict(checkpoint['state_dict'],strict=True)
    with (OUTPUT/'master.csv').open(encoding='utf-8',newline='') as file:
        reader=csv.DictReader(file)
        columns=list(reader.fieldnames)
        rows=list(reader)
    if len(rows)!=12 or len({r['Case_ID'] for r in rows})!=12:
        raise ValueError('Expected the existing 12-case combined table.')
    for row in rows:
        if common.sha256(row['Checkpoint_Path']) != row['Checkpoint_SHA256']:
            raise ValueError(f"Suspect checkpoint changed: {row['Checkpoint_Path']}")
    private_indices=np.load(ROOT/'Indices/CIFAR-10/group_A_subset_10000_from_25000_seed42.npy')[:1000]
    if len(set(private_indices.tolist()))!=1000:
        raise ValueError('Expected 1000 distinct private indices.')
    config=dict(protocol='existing_victim_regressor_legacy_input_v1',
        regressor_path=OLD_REGRESSOR.as_posix(),regressor_sha256=common.sha256(OLD_REGRESSOR),
        trained_in_this_run=False,walk=walk,point_batch_size=8,n_test_per_set=1000,alpha=0.05,
        private_transform='RandomCrop(32,padding=4),RandomHorizontalFlip(p=0.5),ToTensor',
        public_transform='ToTensor',private_indices=private_indices.tolist(),public_indices=list(range(1000)),
        seed_per_suspect=42,regressor_training_split='not recorded in old checkpoint; no disjointness claim',
        implementation_sha256={name:common.sha256(ROOT/name) for name in [Path(__file__).name,
            'AdvAttack/DI.py','Dataset/CIFAR_10.py','util_adv.py']})
    config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',cases=len(rows),config_id=config_id,
            regressor=config['regressor_path'],weights_loaded_strictly=True,training=False,
            walk=walk,private_transform=config['private_transform'],public_transform=config['public_transform']),indent=2))
        return
    json_write=retry_file_write(common.atomic_json)
    metadata_columns=['DI_Regressor_Path','DI_Regressor_SHA256','DI_Config_ID','DI_Input_Protocol']
    columns += [name for name in metadata_columns if name not in columns]
    def write_table():
        tmp=OUTPUT/'master.csv.tmp'
        with tmp.open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(OUTPUT/'master.csv')
    save=retry_file_write(write_table)
    manifest_path=OUTPUT/'existing_di_manifest.json'
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved['config_id']!=config_id:
            raise RuntimeError('Existing DI override has different settings; refusing to mix runs.')
    else:
        archive=OUTPUT/'archive_new_regressor_run'
        archive.mkdir(exist_ok=True)
        for name in ['master.csv','manifest.json','status.json','README.md']:
            src=OUTPUT/name
            dst=archive/name
            if dst.exists():
                if dst.read_bytes()!=src.read_bytes():
                    raise RuntimeError(f'Archive already contains a different {name}.')
            elif src.exists():
                dst.write_bytes(src.read_bytes())
        json_write(manifest_path,dict(config_id=config_id,config=config,
            supersedes='DI columns in archive_new_regressor_run/master.csv',
            other_baselines_provenance='manifest.json'))
    # Reset only DI results that were computed with the superseded regressor.
    for row in rows:
        if row.get('DI_Config_ID')!=config_id:
            for name in columns:
                if name.startswith('DI_'):
                    row[name]=''
            row['DI_Status']='pending'
        row.update(DI_Regressor_Path=config['regressor_path'],DI_Regressor_SHA256=config['regressor_sha256'],
                   DI_Config_ID=config_id,DI_Input_Protocol=config['protocol'])
    save()
    log=(OUTPUT/'run_existing_di.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout,sys.stderr=common.Tee(sys.stdout,log),common.Tee(sys.stderr,log)
    def status(state,**extra):
        json_write(OUTPUT/'status.json',dict(state=state,pid=os.getpid(),baseline='DI',di_config_id=config_id,
            updated_at=datetime.now().isoformat(),total=48,
            completed=sum(r[f'{m}_Status']=='complete' for r in rows for m in common.METHODS),
            failed=sum(r[f'{m}_Status']=='failed' for r in rows for m in common.METHODS),**extra))
    try:
        status('starting')
        print(f'Loading existing regressor only: {OLD_REGRESSOR}',flush=True)
        print(f'Config {config_id}; no training. Private transform: {config["private_transform"]}',flush=True)
        set_seed(42)
        ds,num_classes,_=build_dataset_from_yaml(dict(name='CIFAR-10',normalization='cifar10',
            img_size=32,group_size=25000,download=False))
        # Verification does not need to load/query the victim model again.
        pipeline=DatasetInferencePipeline(None,**walk,point_batch_size=8,device='cuda')
        pipeline.regressor=regressor.to('cuda').eval()
        for row in rows:
            if row['DI_Status']=='complete':
                continue
            model=None
            start=time.monotonic()
            row['DI_Status']='running'
            save()
            status('running',case_id=row['Case_ID'])
            print(f"\n[DI existing regressor] {row['Case_ID']}",flush=True)
            try:
                set_seed(42)
                model=load_positive_suspect('ResNet-18',num_classes,ds,Path(row['Checkpoint_Path']))
                result=pipeline.verify_suspect(model,ds.raw_train_set,ds.raw_test_set,
                    private_indices=private_indices,public_indices=list(range(1000)),n_test_samples=1000,alpha=0.05)
                metric_map={'Mean_Private':'mean_private','Mean_Public':'mean_public','Delta':'delta',
                            'T_Stat':'t_stat','P_Value':'p_value','Stolen':'stolen'}
                metrics={name:int(result[key]) if key=='stolen' else float(result[key]) for name,key in metric_map.items()}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise ValueError(f'Non-finite DI metric: {metrics}')
                row.update({f'DI_{name}':value for name,value in metrics.items()})
                row['DI_Status']='complete'
                row['DI_Error']=''
                print(f'RESULT DI existing: {json.dumps(metrics)}',flush=True)
            except Exception:
                row['DI_Status']='failed'
                row['DI_Error']=traceback.format_exc()
                print(row['DI_Error'],flush=True)
            finally:
                row['DI_Seconds']=round(time.monotonic()-start,3)
                row['DI_Updated_At']=datetime.now().isoformat(timespec='seconds')
                save()
                del model
                torch.cuda.empty_cache()
                status('running',case_id=row['Case_ID'])
        failures=sum(row['DI_Status']!='complete' for row in rows)
        status('completed_with_errors' if failures else 'complete')
        print(f'DI finished: {12-failures}/12 cases. Master: {OUTPUT/"master.csv"}',flush=True)
        if failures:
            raise SystemExit(1)
    except BaseException as error:
        if not isinstance(error,SystemExit):
            status('stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise


if __name__=='__main__':
    main()
