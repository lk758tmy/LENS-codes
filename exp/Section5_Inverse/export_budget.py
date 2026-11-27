"""Export all fixed-budget inverse cases; no training or error-based selection."""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lens
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb
from lens.inverse_data import InverseSource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).parent/'out/budget200_2026-11-19')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'generated')
    parser.add_argument('--time-max', type=float, default=100., help='Displayed time range; retain the full fitting budget')
    parser.add_argument('--report-time', type=float, default=200., help='Table cutoff: last saved feasible state at or before this time')
    parser.add_argument('--no-paper-check', action='store_true')
    args = parser.parse_args(argv)
    if args.time_max<=0 or args.report_time<=0:parser.error('Positive display and report times required')
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.data_root/'protocol.json').read_text())
    rows = [json.loads(p.read_text()) for p in args.data_root.glob('*.json')]
    rows = [r for r in rows if isinstance(r,dict) and 'run_id' in r]
    expected = {(m,s,e,k) for m in ('LENS','FDM') for s in ('A','B') for e in (0.,.01)
                for k in ([0] if e==0 else range(5))}
    actual = {(r['method'],r['source'],r['noise'],r['seed']) for r in rows}
    if actual!=expected or len(rows)!=24:
        raise RuntimeError(f'Expected exactly 24 complete cases; got {len(rows)}; missing {expected-actual}')
    for r in rows:
        if r['budget_seconds']!=protocol['seconds'] or any(v['n_out'] for v in r['curve']):
            raise RuntimeError(f'Inconsistent budget or invalid inversion: {r["run_id"]}')
        if max(v['elapsed'] for v in r['curve'])>protocol['seconds']:
            raise RuntimeError(f'Checkpoint outside budget: {r["run_id"]}')
        if any(p.get('available') and p.get('ac_line_status')!=1 for p in (r['power_before'],r['power_after'])):
            raise RuntimeError(f'Battery timing: {r["run_id"]}')
        with np.load(args.data_root/(r['run_id']+'.npz')) as data:
            np.testing.assert_array_equal(data['eta'],data[f'eta_{len(r["curve"])-1:03d}'])

    report_end=min(args.report_time,protocol['seconds'])
    report_points={}
    for r in rows:
        eligible=[(i,p) for i,p in enumerate(r['curve']) if p['elapsed']<=report_end]
        if not eligible:raise RuntimeError('No feasible saved state before cutoff: '+r['run_id'])
        index,point=eligible[-1]
        report_points[r['run_id']]=dict(index=index,**point)

    def group(method,source,noise):
        return sorted([r for r in rows if (r['method'],r['source'],r['noise'])==(method,source,noise)],
                      key=lambda r:r['seed'])

    def fmt(values):
        values = 100*np.asarray(values)
        return f'{values[0]:.3g}' if len(values)==1 else f'{values.mean():.3g} \\pm {values.std(ddof=1):.2g}'

    table,groups = [],[]
    keys = ('source_E2','source_Einf','E_q_val')
    for source in ('A','B'):
        for noise in (0.,.01):
            for method in ('LENS','FDM'):
                g = group(method,source,noise)
                values = {key:[report_points[r['run_id']][key] for r in g] for key in keys}
                table.append(f'{source} & {100*noise:g} & {method} & '+
                             ' & '.join('$'+fmt(values[key])+'$' for key in keys)+r' \\')
                groups.append(dict(source=source,noise=noise,method=method,count=len(g),
                    **{key:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else 0.) for key,v in values.items()},
                    active_seconds=[r['fit_seconds'] for r in g],iterations=[r['iterations'] for r in g],
                    statuses=[r['status'] for r in g],
                    table_checkpoint_seconds=[report_points[r['run_id']]['elapsed'] for r in g]))
    body = '\n'.join(table)+'\n'
    (args.out/'table_inverse_body.tex').write_text(body,encoding='utf-8')
    compact=lambda value:re.sub(r'\s+','',value).replace(r'\%','')
    if not args.no_paper_check and compact(body) not in compact((ROOT/'report/main25.tex').read_text(encoding='utf-8')):
        raise RuntimeError('Table differs from paper copy; use --no-paper-check for a fresh reproduction')
    summary = dict(data_root=args.data_root.relative_to(ROOT).as_posix() if args.data_root.is_relative_to(ROOT) else str(args.data_root),
                   cases=len(rows),budget_seconds=protocol['seconds'],FDM_n=protocol['FDM_n'],groups=groups)
    summary.update(table_until_seconds=report_end,source_profile_seconds=protocol['seconds'],
        table_selection='last recorded feasible state at or before cutoff; final state held after numerical stop',
        table_checkpoints=report_points)
    audits=json.loads((args.data_root/'final_audit.json').read_text())
    lens_audits=[v for v in audits if v['run_id'].startswith('LENS_')]
    summary['audit']=dict(LENS_training_feasible=sum(v['training_129_grid_feasible'] for v in lens_audits),
        LENS_additional_97_floor01_pass=sum(v['additional_97_grid_floor_01'] for v in lens_audits),
        LENS_additional_97_concave=sum(v['additional_97_strictly_concave'] for v in lens_audits),
        additional_minimum_concavity=-max(max(v['chart_max_wtt_97']) for v in lens_audits),
        max_physical_PDE_residual=max(f['source_pde_max'] for v in lens_audits for f in v['physical_fields']))
    summary['plot_until_seconds']=min(args.time_max,protocol['seconds'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.size':10,'mathtext.fontset':'dejavuserif'})
    colors = {'LENS':'#2166ac','FDM':'#c45d17'}
    budget = protocol['seconds']
    display_end=min(args.time_max,budget)
    dense_path=args.data_root/'fdm_iteration_curves.json'
    dense=json.loads(dense_path.read_text())['curves'] if dense_path.exists() else {}
    grid_all = np.unique(np.r_[0.,budget,*[point['elapsed'] for r in rows for point in r['curve']],
                          *[point['elapsed'] for r in rows if r['method']=='FDM'
                            for point in r['history'] if point['elapsed']<=budget]])
    grid=np.unique(np.r_[0.,display_end,grid_all[grid_all<=display_end]])

    def curve(record,key):
        points = record['curve']
        if record['run_id'] in dense:
            points=dense[record['run_id']]
        elif record['method']=='FDM' and key=='source_E2':
            # The small source vector is present at every outer iteration,
            # including iterations between the full-field checkpoints.
            nodes,weights = np.polynomial.legendre.leggauss(256)
            truth = InverseSource(record['legacy_source'])((nodes+1)/2)
            points = [dict(elapsed=h['elapsed'],source_E2=float(np.sqrt(
                np.sum(weights*(cheb.chebval(nodes,h['eta'])-truth)**2)/np.sum(weights*truth**2))))
                for h in record['history'] if h['elapsed']<=budget]
        times = np.array([point['elapsed'] for point in points])
        values = np.array([point[key] for point in points])
        indices = np.maximum(np.searchsorted(times,grid,side='right')-1,0)
        return 100*values[indices]

    fig,axes = plt.subplots(2,2,figsize=(7.0,4.8),sharex=True,constrained_layout=False)
    handles,labels = [],[]
    for col,source in enumerate(('A','B')):
        axes[0,col].set_title(f'Source {source}')
        for row,key in enumerate(('source_E2','E_q_val')):
            ax = axes[row,col]
            for method in ('LENS','FDM'):
                for noise in (0.,.01):
                    values = np.array([curve(r,key) for r in group(method,source,noise)])
                    line, = ax.step(grid,values.mean(0),where='post',color=colors[method],
                                    ls='-' if noise==0 else '--',lw=1.3,
                                    label=f'{method}, '+('no noise' if noise==0 else '1% noise'))
                    if len(values)>1:
                        ax.fill_between(grid,values.min(0),values.max(0),step='post',
                                        color=colors[method],alpha=.13,lw=0)
                    if row==col==0:
                        handles.append(line);labels.append(line.get_label())
            ax.set_yscale('log')
            ax.set_xlim(0,display_end)
            ax.set_ylabel(r'$E_2^f$ (%)' if key=='source_E2' else r'$E_q^{\mathrm{val}}$ (%)')
            ax.grid(alpha=.22,which='major')
            if row==1:ax.set_xlabel('Elapsed time (s)')
    fig.subplots_adjust(left=.105,right=.96,bottom=.18,top=.94,wspace=.29,hspace=.17)
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,
               frameon=False,fontsize=8,columnspacing=1.2,handlelength=2.4)
    fig.savefig(args.out/'inverse_error_time.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(args.out/'inverse_error_time.png',dpi=200)
    plt.close(fig)

    # Alternative layout only: the existing 2x2 plot is retained unchanged.
    fig,axes=plt.subplots(1,4,figsize=(12.0,2.95))
    for panel,(source,key) in enumerate((('A','source_E2'),('B','source_E2'),
                                         ('A','E_q_val'),('B','E_q_val'))):
        ax=axes[panel]
        ax.set_title(f'({chr(97+panel)}) Source {source}')
        for method in ('LENS','FDM'):
            for noise in (0.,.01):
                values=np.array([curve(r,key) for r in group(method,source,noise)])
                ax.step(grid,values.mean(0),where='post',color=colors[method],
                        ls='-' if noise==0 else '--',lw=1.3,
                        label=f'{method}, '+('no noise' if noise==0 else '1% noise'))
                if len(values)>1:
                    ax.fill_between(grid,values.min(0),values.max(0),step='post',
                                    color=colors[method],alpha=.13,lw=0)
        ax.set_yscale('log')
        ax.set_xlim(0,display_end)
        ax.set_xlabel('Elapsed time (s)')
        ax.set_ylabel(r'$E_2^f$ (%)' if key=='source_E2' else r'$E_q^{\mathrm{val}}$ (%)')
        ax.grid(alpha=.22,which='major')
    fig.subplots_adjust(left=.065,right=.99,bottom=.29,top=.87,wspace=.43)
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,
               frameon=False,fontsize=9,columnspacing=1.5,handlelength=2.4)
    fig.savefig(args.out/'inverse_error_time_horizontal.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(args.out/'inverse_error_time_horizontal.png',dpi=200)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(7.,2.8))
    for ax,source in zip(axes,('A','B')):
        initial = np.load(args.data_root/(group('LENS',source,0)[0]['run_id']+'.npz'))
        x = initial['x']
        ax.plot(x,initial['true_source'],color='black',lw=1.5,label='True source',zorder=2)
        for method in ('LENS',):
            for noise in (0.,.01):
                values = np.array([np.load(args.data_root/(r['run_id']+'.npz'))['source'] for r in group(method,source,noise)])
                ax.plot(x,values.mean(0),color=colors[method],ls='-' if noise==0 else '--',lw=.7,
                        label=f'{method}, '+('no noise' if noise==0 else '1% noise'))
                if len(values)>1:
                    ax.fill_between(x,values.min(0),values.max(0),color=colors[method],alpha=.13,lw=0)
        ax.set(xlabel='$x$',ylabel='$f(x)$',title=f'Source {source}',xlim=(0,1))
        ax.grid(alpha=.2)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=.085,right=.96,bottom=.23,top=.89,wspace=.25)
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=3,frameon=False,fontsize=8,
               columnspacing=1.0,handlelength=2.3,handletextpad=.5)
    fig.canvas.draw()
    legend_box=legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert 0<=legend_box.x0<legend_box.x1<=1 and 0<=legend_box.y0<legend_box.y1<=.12, legend_box
    fig.savefig(args.out/'inverse_sources.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(args.out/'inverse_sources.png',dpi=200)
    plt.close(fig)

    lines = [f'# 反问题比较：{report_end:g}秒表格与{budget:g}秒原始运行','',
        '两源、两方法、无噪声1次及1%噪声5次，共24组。所有种子均计入结果。表内误差为百分数；±为样本标准差。',
        '', '同一源使用相同观测和噪声；源空间均为16阶Chebyshev多项式，初值为1。LENS最终K=24，FDM n='+str(protocol['FDM_n'])+'，独立参考n=640。',
        '无噪声rho=0，1%噪声统一固定rho=1e-9。正式运行不扫描rho，不以真解或验证误差选择迭代点。',
        f'时间包含各自构建、初始化和训练，排除共享数据读取、模块导入、保存和后验评价。原始运行保留{budget:g}秒内完整求出的最后有效解。表格采用{report_end:g}秒前最后保存的有效状态；达到数值停止条件后保留最终值。',
        f'误差曲线仅展示前{display_end:g}秒，完整200秒原始记录及旧表格另存保留；源重建图仍为200秒终值。记录点间为前值保持，提前数值停止后保持末值。1%噪声曲线为5种子的均值，阴影为范围。验证响应只在拟合时钟停止后求值。',
        '', '|源|噪声 %|方法|E2f %|Einff %|Eqval %|实际训练 s|步数|停止原因|',
        '|---|---:|---|---:|---:|---:|---|---|---|']
    for g in groups:
        values = [fmt([report_points[r['run_id']][key] for r in group(g['method'],g['source'],g['noise'])]).replace('\\pm','±') for key in keys]
        lines.append(f"|{g['source']}|{100*g['noise']:g}|{g['method']}|"+'|'.join(values)+
                     '|'+', '.join(f'{v:.2f}' for v in g['active_seconds'])+'|'+str(g['iterations'])+'|'+str(g['statuses'])+'|')
    lines += ['', '表格快照时刻（秒，按种子排序）：'+json.dumps({g['method']+'_'+g['source']+'_e'+str(g['noise']):g['table_checkpoint_seconds'] for g in groups}),
              '', '提前停止和时间上限均不是源误差达到某个阈值的声明。不同方法的离散空间和每步工作量不同，不将外迭代数当成相同计算量。',
              f"原200秒终态的129×129训练检查通过 {summary['audit']['LENS_training_feasible']}/12；额外97×97网格上的0.1目标余量通过 {summary['audit']['LENS_additional_97_floor01_pass']}/12，严格负曲率符号通过 {summary['audit']['LENS_additional_97_concave']}/12。额外网格最小 -omega_tt={summary['audit']['additional_minimum_concavity']:.6g}，物理网格最大PDE残差={summary['audit']['max_physical_PDE_residual']:.6g}。这些检查不是连续证书。",
              '原有release_v6及其rho路径完整保留，但不再用作本轮正文主表。']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
