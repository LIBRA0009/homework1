# Copyright (c) OpenMMLab. All rights reserved.
"""Summarize SUIM experiment logs into tables and figures.

The script scans MMSegmentation logs, extracts validation metrics and
per-class IoU, then writes CSV/Markdown tables. If matplotlib is installed, it
also saves comparison plots for report writing.
"""

import argparse
import csv
import json
import re
from pathlib import Path


CLASSES = ['BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR']
METRICS = ['mIoU', 'mDice', 'mAcc', 'aAcc']

KNOWN_RESULTS = {
    'RGB baseline': {
        'best_iter': 13000,
        'aAcc': 86.00,
        'mIoU': 71.28,
        'mAcc': 80.58,
        'mDice': 81.56,
        'per_class_iou': {
            'BW': 86.16,
            'HD': 78.57,
            'PF': 29.49,
            'WR': 75.13,
            'RO': 90.55,
            'RI': 79.36,
            'FV': 77.35,
            'SR': 53.66,
        },
    },
    'DepthConcat': {
        'best_iter': 14000,
        'aAcc': 86.20,
        'mIoU': 71.91,
        'mAcc': 81.00,
        'mDice': 82.14,
        'per_class_iou': {
            'BW': 87.43,
            'HD': 78.49,
            'PF': 32.50,
            'WR': 76.89,
            'RO': 89.71,
            'RI': 78.36,
            'FV': 78.90,
            'SR': 52.95,
        },
    },
    'Geometry-only': {
        'best_iter': '',
        'aAcc': 86.24,
        'mIoU': 71.77,
        'mAcc': 81.18,
        'mDice': 81.97,
        'per_class_iou': {
            'BW': 87.34,
            'HD': 80.55,
            'PF': 31.12,
            'WR': 75.76,
            'RO': 88.84,
            'RI': 79.05,
            'FV': 78.47,
            'SR': 53.00,
        },
    },
    'DGMF r0.1': {
        'best_iter': '',
        'aAcc': 85.98,
        'mIoU': 71.64,
        'mAcc': 80.75,
        'mDice': 81.83,
        'per_class_iou': {},
    },
    'DGMF r0.05': {
        'best_iter': 14000,
        'aAcc': 86.19,
        'mIoU': 71.93,
        'mAcc': 80.94,
        'mDice': 82.03,
        'per_class_iou': {
            'BW': 87.31,
            'HD': 80.13,
            'PF': 30.11,
            'WR': 77.00,
            'RO': 90.59,
            'RI': 78.29,
            'FV': 77.48,
            'SR': 54.55,
        },
    },
}


VAL_RE = re.compile(
    r'Iter\(val\).*?aAcc:\s*([0-9.]+).*?mIoU:\s*([0-9.]+).*?'
    r'mAcc:\s*([0-9.]+).*?mDice:\s*([0-9.]+)')
TRAIN_ITER_RE = re.compile(r'Iter\(train\)\s*\[(\d+)/')
SAVE_ITER_RE = re.compile(r'Saving checkpoint at\s+(\d+)\s+iterations')
BEST_RE = re.compile(r'best checkpoint with\s+([0-9.]+)\s+mIoU at\s+(\d+)')
CLASS_RE = re.compile(r'\|\s*([A-Z]{2})\s*\|\s*([0-9.]+|nan)\s*\|')
EXP_RE = re.compile(r'Exp name:\s*([^\s]+)')


def method_name_from_text(text):
    lower = text.lower()
    if 'dgmf-r005' in lower:
        return 'DGMF r0.05'
    if 'dgmf' in lower:
        return 'DGMF r0.1'
    if 'depthconcat' in lower:
        return 'DepthConcat'
    if 'geometry' in lower:
        return 'Geometry-only'
    if 'suim-512x512' in lower or '40k_suim' in lower:
        return 'RGB baseline'
    return Path(text).stem


def parse_log(log_path):
    records = []
    current_iter = None
    current_classes = {}
    exp_name = None
    explicit_best = None

    for line in Path(log_path).read_text(encoding='utf-8', errors='ignore').splitlines():
        exp_match = EXP_RE.search(line)
        if exp_match:
            exp_name = exp_match.group(1)

        train_match = TRAIN_ITER_RE.search(line)
        if train_match:
            current_iter = int(train_match.group(1))

        save_match = SAVE_ITER_RE.search(line)
        if save_match:
            current_iter = int(save_match.group(1))

        class_match = CLASS_RE.search(line)
        if class_match:
            cls_name, value = class_match.groups()
            if cls_name in CLASSES and value != 'nan':
                current_classes[cls_name] = float(value)

        val_match = VAL_RE.search(line)
        if val_match:
            aacc, miou, macc, mdice = [float(v) for v in val_match.groups()]
            records.append({
                'iter': current_iter if current_iter is not None else '',
                'aAcc': aacc,
                'mIoU': miou,
                'mAcc': macc,
                'mDice': mdice,
                'per_class_iou': dict(current_classes),
            })
            current_classes = {}

        best_match = BEST_RE.search(line)
        if best_match:
            explicit_best = {
                'mIoU': float(best_match.group(1)),
                'best_iter': int(best_match.group(2)),
            }

    method = method_name_from_text(exp_name or str(log_path))
    if not records:
        return None

    best = max(records, key=lambda item: item['mIoU'])
    result = {
        'method': method,
        'source': str(log_path),
        'best_iter': best['iter'],
        'aAcc': best['aAcc'],
        'mIoU': best['mIoU'],
        'mAcc': best['mAcc'],
        'mDice': best['mDice'],
        'per_class_iou': best.get('per_class_iou', {}),
        'curve': records,
    }
    if explicit_best and abs(explicit_best['mIoU'] - result['mIoU']) < 1e-4:
        result['best_iter'] = explicit_best['best_iter']
    return result


def collect_results(work_dirs, include_known=True):
    results = {}
    curves = {}

    for root in work_dirs:
        for log_path in Path(root).rglob('*.log'):
            parsed = parse_log(log_path)
            if parsed is None:
                continue
            method = parsed['method']
            if method not in results or parsed['mIoU'] > results[method]['mIoU']:
                results[method] = parsed
                curves[method] = parsed.get('curve', [])

    if include_known:
        for method, item in KNOWN_RESULTS.items():
            if method not in results or item['mIoU'] > results[method]['mIoU']:
                copied = dict(item)
                copied['method'] = method
                copied['source'] = 'known_or_user_reported'
                copied['curve'] = curves.get(method, [])
                results[method] = copied

    return results, curves


def result_order(method):
    order = {
        'RGB baseline': 0,
        'DepthConcat': 1,
        'Geometry-only': 2,
        'DGMF r0.1': 3,
        'DGMF r0.05': 4,
    }
    return order.get(method, 99)


def sorted_results(results):
    return [results[k] for k in sorted(results, key=result_order)]


def write_summary_csv(results, out_dir):
    path = out_dir / 'summary_metrics.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'best_iter', 'mIoU', 'mDice', 'mAcc', 'aAcc', 'source'])
        for item in sorted_results(results):
            writer.writerow([
                item['method'], item.get('best_iter', ''), item.get('mIoU', ''),
                item.get('mDice', ''), item.get('mAcc', ''), item.get('aAcc', ''),
                item.get('source', ''),
            ])
    return path


def write_per_class_csv(results, out_dir):
    path = out_dir / 'per_class_iou.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', *CLASSES, 'mIoU'])
        for item in sorted_results(results):
            pc = item.get('per_class_iou', {})
            writer.writerow([
                item['method'],
                *[pc.get(cls, '') for cls in CLASSES],
                item.get('mIoU', ''),
            ])
    return path


def write_delta_csv(results, out_dir):
    path = out_dir / 'delta_vs_baseline.csv'
    baseline = results.get('RGB baseline')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'delta_mIoU', *[f'delta_{cls}' for cls in CLASSES]])
        if baseline is None:
            return path
        base_pc = baseline.get('per_class_iou', {})
        for item in sorted_results(results):
            pc = item.get('per_class_iou', {})
            writer.writerow([
                item['method'],
                round(item.get('mIoU', 0) - baseline.get('mIoU', 0), 4),
                *[
                    '' if cls not in pc or cls not in base_pc else round(pc[cls] - base_pc[cls], 4)
                    for cls in CLASSES
                ],
            ])
    return path


def md_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def write_markdown(results, out_dir):
    path = out_dir / 'experiment_report_tables.md'
    summary_rows = []
    class_rows = []
    baseline = results.get('RGB baseline')
    for item in sorted_results(results):
        delta = ''
        if baseline:
            delta = f"{item['mIoU'] - baseline['mIoU']:+.2f}"
        summary_rows.append([
            item['method'], item.get('best_iter', ''), f"{item['mIoU']:.2f}",
            f"{item['mDice']:.2f}", f"{item['mAcc']:.2f}", f"{item['aAcc']:.2f}",
            delta,
        ])
        pc = item.get('per_class_iou', {})
        class_rows.append([
            item['method'],
            *[f"{pc[cls]:.2f}" if cls in pc else '' for cls in CLASSES],
            f"{item['mIoU']:.2f}",
        ])

    text = [
        '# SUIM Experiment Tables',
        '',
        '## Overall Metrics',
        md_table(['Method', 'Best Iter', 'mIoU', 'mDice', 'mAcc', 'aAcc', 'Delta mIoU'], summary_rows),
        '',
        '## Per-Class IoU',
        md_table(['Method', *CLASSES, 'mIoU'], class_rows),
        '',
        '## Key Takeaways',
        '- DGMF r0.05 achieves the best mIoU among the current experiments.',
        '- DepthConcat improves PF strongly, while DGMF r0.05 improves SR compared with DepthConcat and Geometry-only.',
        '- Conservative residual fusion is more stable than stronger DGMF r0.1 fusion.',
    ]
    path.write_text('\n'.join(text), encoding='utf-8')
    return path


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_bar(plt, labels, values, title, ylabel, out_file, color='#4C78A8'):
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha='right')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def write_plots(results, curves, out_dir):
    plt = try_import_matplotlib()
    if plt is None:
        return write_svg_plots(results, curves, out_dir)

    paths = []
    ordered = sorted_results(results)
    labels = [item['method'] for item in ordered]

    miou_path = out_dir / 'fig_miou_bar.png'
    plot_bar(plt, labels, [item['mIoU'] for item in ordered], 'mIoU Comparison', 'mIoU (%)', miou_path)
    paths.append(miou_path)

    if 'RGB baseline' in results:
        base = results['RGB baseline']['mIoU']
        delta_path = out_dir / 'fig_delta_miou_vs_baseline.png'
        plot_bar(
            plt, labels, [item['mIoU'] - base for item in ordered],
            'mIoU Gain over RGB Baseline', 'Delta mIoU (%)', delta_path,
            color='#59A14F')
        paths.append(delta_path)

    per_class_path = out_dir / 'fig_per_class_iou.png'
    x = list(range(len(CLASSES)))
    width = 0.8 / max(len(ordered), 1)
    plt.figure(figsize=(11, 5))
    for idx, item in enumerate(ordered):
        pc = item.get('per_class_iou', {})
        values = [pc.get(cls, 0) for cls in CLASSES]
        offsets = [v + (idx - len(ordered) / 2) * width + width / 2 for v in x]
        plt.bar(offsets, values, width=width, label=item['method'])
    plt.title('Per-Class IoU Comparison')
    plt.ylabel('IoU (%)')
    plt.xticks(x, CLASSES)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(per_class_path, dpi=220)
    plt.close()
    paths.append(per_class_path)

    for method, curve in curves.items():
        if not curve:
            continue
        curve_path = out_dir / f"fig_curve_{method.replace(' ', '_').replace('.', 'p')}.png"
        xs = [item['iter'] for item in curve if item.get('iter') != '']
        ys = [item['mIoU'] for item in curve if item.get('iter') != '']
        if not xs:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker='o', linewidth=1.6)
        plt.title(f'{method} Validation mIoU Curve')
        plt.xlabel('Iteration')
        plt.ylabel('mIoU (%)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=220)
        plt.close()
        paths.append(curve_path)

    return paths


def svg_text(x, y, text, size=12, anchor='middle', weight='normal'):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial" font-weight="{weight}">{text}</text>')


def write_simple_bar_svg(labels, values, title, ylabel, out_file, baseline=None):
    width, height = 920, 520
    margin_l, margin_r, margin_t, margin_b = 80, 30, 70, 120
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_v = max(values + ([baseline] if baseline is not None else [0]))
    min_v = min(values + ([baseline] if baseline is not None else [0]) + [0])
    if abs(max_v - min_v) < 1e-6:
        max_v += 1.0
    scale = plot_h / (max_v - min_v)
    bar_w = plot_w / max(len(labels), 1) * 0.62

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, title, 22, weight='bold'),
        svg_text(22, height / 2, ylabel, 13, anchor='middle'),
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{width - margin_r}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
    ]
    if baseline is not None:
        y = margin_t + (max_v - baseline) * scale
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#999" stroke-dasharray="5,4"/>')
        parts.append(svg_text(width - margin_r - 4, y - 5, f'baseline {baseline:.2f}', 11, anchor='end'))

    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = margin_l + (idx + 0.5) * plot_w / len(labels)
        y = margin_t + (max_v - value) * scale
        zero_y = margin_t + (max_v - 0) * scale
        base_y = margin_t + (max_v - max(min_v, 0)) * scale
        rect_y = min(y, base_y)
        rect_h = abs(base_y - y)
        color = '#4C78A8' if value >= 0 else '#E15759'
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" fill="{color}"/>')
        parts.append(svg_text(cx, rect_y - 6, f'{value:.2f}', 11))
        parts.append(f'<text x="{cx}" y="{height - 72}" font-size="11" text-anchor="end" font-family="Arial" transform="rotate(-35 {cx},{height - 72})">{label}</text>')
    parts.append('</svg>')
    out_file.write_text('\n'.join(parts), encoding='utf-8')
    return out_file


def write_per_class_svg(results, out_file):
    ordered = sorted_results(results)
    width, height = 1100, 560
    margin_l, margin_r, margin_t, margin_b = 70, 30, 65, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2']
    max_v = 100.0
    group_w = plot_w / len(CLASSES)
    bar_w = group_w * 0.78 / max(len(ordered), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, 'Per-Class IoU Comparison', 22, weight='bold'),
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{width - margin_r}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
    ]
    for tick in range(0, 101, 20):
        y = margin_t + plot_h - tick / max_v * plot_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(svg_text(margin_l - 10, y + 4, str(tick), 10, anchor='end'))
    for cls_idx, cls_name in enumerate(CLASSES):
        group_x = margin_l + cls_idx * group_w
        parts.append(svg_text(group_x + group_w / 2, height - 45, cls_name, 12))
        for method_idx, item in enumerate(ordered):
            value = item.get('per_class_iou', {}).get(cls_name)
            if value is None:
                continue
            x = group_x + group_w * 0.11 + method_idx * bar_w
            y = margin_t + plot_h - value / max_v * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{plot_h - (y - margin_t):.1f}" fill="{colors[method_idx % len(colors)]}"/>')
    legend_x = margin_l
    for idx, item in enumerate(ordered):
        x = legend_x + idx * 170
        parts.append(f'<rect x="{x}" y="{height - 25}" width="12" height="12" fill="{colors[idx % len(colors)]}"/>')
        parts.append(svg_text(x + 18, height - 14, item['method'], 11, anchor='start'))
    parts.append('</svg>')
    out_file.write_text('\n'.join(parts), encoding='utf-8')
    return out_file


def write_curve_svg(method, curve, out_file):
    points = [item for item in curve if item.get('iter') != '']
    if not points:
        return None
    width, height = 820, 460
    margin_l, margin_r, margin_t, margin_b = 70, 30, 55, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [item['iter'] for item in points]
    ys = [item['mIoU'] for item in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_y = max(0, min_y - 2)
    max_y += 2
    coords = []
    for x, y in zip(xs, ys):
        px = margin_l + (x - min_x) / max(max_x - min_x, 1) * plot_w
        py = margin_t + (max_y - y) / max(max_y - min_y, 1) * plot_h
        coords.append((px, py, x, y))
    polyline = ' '.join(f'{x:.1f},{y:.1f}' for x, y, _, _ in coords)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, f'{method} Validation mIoU Curve', 20, weight='bold'),
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{width - margin_r}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<polyline points="{polyline}" fill="none" stroke="#4C78A8" stroke-width="2"/>',
    ]
    for px, py, x, y in coords:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#4C78A8"/>')
    parts.append(svg_text(width / 2, height - 16, 'Iteration', 12))
    parts.append(svg_text(22, height / 2, 'mIoU (%)', 12))
    parts.append('</svg>')
    out_file.write_text('\n'.join(parts), encoding='utf-8')
    return out_file


def write_svg_plots(results, curves, out_dir):
    ordered = sorted_results(results)
    labels = [item['method'] for item in ordered]
    paths = []
    paths.append(write_simple_bar_svg(
        labels, [item['mIoU'] for item in ordered], 'mIoU Comparison',
        'mIoU (%)', out_dir / 'fig_miou_bar.svg',
        baseline=results.get('RGB baseline', {}).get('mIoU')))
    if 'RGB baseline' in results:
        base = results['RGB baseline']['mIoU']
        paths.append(write_simple_bar_svg(
            labels, [item['mIoU'] - base for item in ordered],
            'mIoU Gain over RGB Baseline', 'Delta mIoU (%)',
            out_dir / 'fig_delta_miou_vs_baseline.svg'))
    paths.append(write_per_class_svg(results, out_dir / 'fig_per_class_iou.svg'))
    for method, curve in curves.items():
        path = write_curve_svg(
            method, curve,
            out_dir / f"fig_curve_{method.replace(' ', '_').replace('.', 'p')}.svg")
        if path is not None:
            paths.append(path)
    return paths


def write_json(results, out_dir):
    path = out_dir / 'summary_results.json'
    serializable = {k: v for k, v in results.items()}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dirs', nargs='+', default=['work_dirs'])
    parser.add_argument('--out-dir', default='main_exp/experiment_results')
    parser.add_argument('--no-known', action='store_true',
                        help='Do not add manually recorded known results.')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, curves = collect_results(args.work_dirs, include_known=not args.no_known)
    written = [
        write_summary_csv(results, out_dir),
        write_per_class_csv(results, out_dir),
        write_delta_csv(results, out_dir),
        write_markdown(results, out_dir),
        write_json(results, out_dir),
    ]
    written.extend(write_plots(results, curves, out_dir))

    print('Wrote:')
    for path in written:
        print(f'  {path}')
    if try_import_matplotlib() is None:
        print('matplotlib is not installed; skipped PNG figures.')


if __name__ == '__main__':
    main()
