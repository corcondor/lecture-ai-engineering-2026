"""Rescore two actual Kaggle outputs using the original course scorers."""
import argparse
import ast
import json
from pathlib import Path
import re
import tempfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs_root',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    notebook=root.parents[2]/'notebooks/lora_format_tuning.ipynb'
    raw_notebook=notebook.read_bytes()
    import hashlib
    assert hashlib.sha1(b'blob '+str(len(raw_notebook)).encode()+b'\0'+raw_notebook).hexdigest()=='f395c4900558995276160f7849648edb98142009'
    nb=json.loads(raw_notebook.decode('utf-8'))
    source=lambda i:''.join(nb['cells'][i]['source'])
    env={'json':json}
    data_nodes=[n for n in ast.parse(source(7)).body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_' for t in n.targets))]
    exec(compile(ast.Module(body=data_nodes,type_ignores=[]),'official_data','exec'),env)
    scorer_nodes=[n for n in ast.parse(source(10)).body if isinstance(n,ast.FunctionDef) and n.name in {'check_format','check_content'}]
    exec(compile(ast.Module(body=scorer_nodes,type_ignores=[]),'official_scorers','exec'),env)
    checked=0
    runs=[]
    for epochs in [20,10]:
        directory=args.runs_root/f'epochs-{epochs}'
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        assert result['executed_training_myself'] is True
        assert result['training_metrics']['global_step']==epochs*2
        assert result['runtime']['gpu']=='Tesla T4'
        evaluations={}
        for phase in ['before','after']:
            evaluation=json.loads((directory/f'eval_{phase}.json').read_text(encoding='utf-8'))
            evaluations[phase]=evaluation
            assert [r['question'] for r in evaluation['rows']]==[e['question'] for e in env['test']]
            for row in evaluation['rows']:
                fmt,reason,obj=env['check_format'](row['raw'])
                loose=obj
                if not fmt:
                    match=re.search(r'\{.*\}',row['raw'],re.S)
                    try: loose=json.loads(match.group(0)) if match else None
                    except Exception: loose=None
                content=env['check_content'](loose if isinstance(loose,dict) else None,env['gold'][row['question']])
                assert row['format_ok']==fmt and row['format_reason']==reason and row['content_ok']==content and row['both_ok']==(fmt and content)
                assert row['gold_needs_review']==env['gold'][row['question']]['needs_review']
                checked+=1
            for key,field in [('format_rate','format_ok'),('content_rate','content_ok'),('both_rate','both_ok')]:
                rate=sum(r[field] for r in evaluation['rows'])/len(evaluation['rows'])
                assert rate==evaluation['summary'][key]==result[phase][key]
        regressed=[b['question'] for b,a in zip(evaluations['before']['rows'],evaluations['after']['rows']) if b['both_ok'] and not a['both_ok']]
        assert regressed==result['regressed']
        losses=[h for h in json.loads((directory/'training_log_history.json').read_text(encoding='utf-8')) if 'loss' in h]
        assert result['training_metrics']['final_logged_loss']==losses[-1]['loss']
        assert result['training_metrics']['final_logged_loss_step']==losses[-1]['step']==epochs*2
        for row in result['knowledge']:
            for phase in ['学習前','学習後']:
                fmt,_,obj=env['check_format'](row[f'{phase}の出力'])
                classification='format_error' if not fmt else ('needs_review' if obj['needs_review'] else 'asserted')
                assert row[f'{phase}の分類']==classification
        runs.append(result)
    diff={k:[runs[0]['config'].get(k),runs[1]['config'].get(k)] for k in set(runs[0]['config'])|set(runs[1]['config']) if runs[0]['config'].get(k)!=runs[1]['config'].get(k)}
    assert diff=={'num_train_epochs':[20,10]}
    assert (args.runs_root/'epochs-20/test.jsonl').read_bytes()==(args.runs_root/'epochs-10/test.jsonl').read_bytes()
    evidence={'actual_evaluation_rows_rescored':checked,'actual_knowledge_outputs_classified':16,'two_training_runs_completed':True,'only_config_difference':diff,'test_data_identical':True,'final_loss_matches_training_history':True,'gpu':'Tesla T4'}
    if args.report:
        args.report.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(evidence))

if __name__=='__main__':main()
