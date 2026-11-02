"""Run one official Day2 configuration in a fresh Kaggle GPU process."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, choices=[20,10], required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    work = root / f'epochs-{args.epochs}'
    work.mkdir(parents=True, exist_ok=True)
    if (work/'result.json').exists():
        raise RuntimeError('Completed result already exists. Preserve the result and use a fresh run folder.')
    nb = json.loads((root/'official.ipynb').read_text(encoding='utf-8'))
    source = lambda i: ''.join(nb['cells'][i]['source'])
    env = {'display':print}
    print('START', args.epochs, time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
    try:
        import torch
        import transformers, peft, trl, accelerate, datasets
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU allocated; no model loading or training started.')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        runtime = {'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                   'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
                   'versions':{m.__name__:m.__version__ for m in [transformers,peft,trl,accelerate,datasets]}}
        (work/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
        print('RUNTIME',json.dumps(runtime),flush=True)
        expected = {'transformers':'5.18.0','peft':'0.21.1','trl':'1.14.2','accelerate':'1.15.0','datasets':'5.1.0'}
        for name, version in expected.items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError(f'{name} must be {version}')
        env.update(torch=torch,transformers=transformers,peft=peft,trl=trl)
        # The reviewed configuration cell creates its work directory before any data/model is loaded.
        config_cell = source(5).replace('"work_dir": "/content/qlora_lab"', '"work_dir": '+repr(str(work)))
        config_cell = config_cell.replace('"num_train_epochs": 20,','"num_train_epochs": '+str(args.epochs)+',',1)
        exec(compile(config_cell,'official_config','exec'),env)
        if env['CONFIG']['work_dir'] != str(work) or env['CONFIG']['num_train_epochs'] != args.epochs:
            raise RuntimeError('Configuration substitution failed')
        for i in [7,9,10]:
            print('CELL',i,flush=True)
            exec(compile(source(i),f'official_cell_{i}','exec'),env)
        training = source(12).replace('trainer.train()  #','trainer.train()\nTRAINING_SECONDS = time.time() - t0\n#',1)
        print('TRAINING',args.epochs,flush=True)
        exec(compile(training,'official_training','exec'),env)
        for i in [15,17,19]:
            print('CELL',i,flush=True)
            exec(compile(source(i),f'official_cell_{i}','exec'),env)
        result = env['result']
        losses = [r for r in env['trainer'].state.log_history if 'loss' in r]
        result['training_metrics'] = {'training_seconds':env['TRAINING_SECONDS'],
             'final_logged_loss':losses[-1]['loss'] if losses else None,
             'final_logged_loss_step':losses[-1]['step'] if losses else None,
             'global_step':env['trainer'].state.global_step,
             'loss_definition':'Last logged training loss, averaged over the final logging interval.'}
        result['runtime'] = runtime
        result['provenance'] = {'kind':'actual_kaggle_training','official_commit':'fae899a409e2c99feee9d44a3789f936f5fc64ee',
             'requested_changed_setting':'num_train_epochs','run_epochs':args.epochs,
             'common_adaptations':['Kaggle output directory','Only first allocated GPU visible','CPU threads 2'],
             'notes':'No course model, data, batch size, learning rate, rank, generation, or evaluation criteria were replaced.'}
        (work/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        (work/'training_log_history.json').write_text(json.dumps(env['trainer'].state.log_history,indent=2),encoding='utf-8')
        print('COMPLETE',args.epochs,json.dumps(result['training_metrics']),flush=True)
    except Exception as exc:
        (work/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'training_completed':False},ensure_ascii=False,indent=2),encoding='utf-8')
        raise

if __name__ == '__main__':
    main()
