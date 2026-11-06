"""Export browser-saved Springer HTML, with optional satellite table recovery."""
import argparse
import json
from collections import Counter
from app.ingestion.loader import load_html_bundle
from app.ingestion.adapters.springer import parse_springer_bundle
from app.ingestion.exporter import export_paper


def main():
    parser=argparse.ArgumentParser(description='Export saved Springer article to PaperDocument V2')
    parser.add_argument('html_path')
    parser.add_argument('--resources',default=None,help='Matching saved _files folder')
    parser.add_argument('--table-file',action='append',default=[],help='Extra saved Springer table page (repeatable)')
    parser.add_argument('--fetch-tables',action='store_true',help='Opt-in direct HTTP requests (may fail with access restrictions)')
    parser.add_argument('--browser-tables',action='store_true',help='Opt-in Microsoft Edge window to fetch missing table pages; requires uv add playwright')
    parser.add_argument('--browser-figures',action='store_true',help='Opt-in Edge recovery of full-size article figures; requires uv add playwright')
    parser.add_argument('--browser-assets',action='store_true',help='Recover both tables and figures using Edge (opt-in)')
    parser.add_argument('--table-dir',action='append',default=[],help='Extra directory of saved Table pages (repeatable)')
    parser.add_argument('--output-root',default='data/standardized',help='Output root folder for exported paper')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    bundle=load_html_bundle(args.html_path,args.resources)
    doc=parse_springer_bundle(bundle,fetch_tables=args.fetch_tables,table_files=args.table_file + args.table_dir, browser_tables=(args.browser_tables or args.browser_assets),
                              browser_figures=(args.browser_figures or args.browser_assets))
    report=export_paper(doc,bundle,args.output_root,overwrite=args.overwrite)
    data=json.loads(report.paper_json.read_text(encoding='utf-8'))
    counts=Counter(x.kind for x in doc.elements.values())
    print('=== SPRINGER DOCUMENT V2 ===')
    print('Paper:',doc.paper_id,'\nTitle:',doc.metadata.title)
    print('Sections:',len(doc.sections),'Elements:',dict(counts))
    print('References:',len(doc.references),'Mentions:',len(doc.mentions))
    print('Resolved:',sum(m.resolution_status=='resolved' for m in doc.mentions),'/',len(doc.mentions))
    print('Tables recovered:',sum(e.kind=='table' and e.grid and len(e.grid.cells)>0 for e in doc.elements.values()),'/',counts['table'])
    preview_figs={w.split(': ',1)[1].split(';',1)[0] for w in doc.quality.warnings
                  if w.startswith('Springer figure preview-only: ')}
    figures_highres=sum(e.kind=='figure' and e.original_html_id not in preview_figs
                       and e.image_asset_id and data['assets'].get(e.image_asset_id,{}).get('status')=='available'
                       for e in doc.elements.values())
    print('Figures higher-resolution confirmed:',figures_highres,'/',counts['figure'])
    print('Assets copied:',report.copied_assets,'Missing:',report.missing_assets,'Skipped:',report.skipped_assets)
    print('Quality (exported):',{k:v for k,v in data['quality'].items() if k!='warnings'})
    print('Reader:',report.normalized_html,'\nJSON:',report.paper_json)
    warnings=data['quality']['warnings']
    for w in warnings:
        if w.startswith(('Table ', 'Edge table', 'Edge figure', 'Figure ', 'Springer figure', 'Saved Springer', 'Explicit Springer')):
            print('Warning:',w)
    other = [w for w in warnings if not w.startswith(('Table ', 'Edge table', 'Edge figure', 'Figure ', 'Springer figure', 'Saved Springer', 'Explicit Springer'))]
    for w in other[:10]:
        print('Warning:',w)
    if counts['table'] and sum(1 for e in doc.elements.values() if e.kind=='table' and e.grid and e.grid.cells) < counts['table']:
        print('INFO: Remaining tables are NOT recovered. Use --browser-tables (uv add playwright)')
        print('      or --table-file/--table-dir to supply saved satellite HTML. Missing is not complete.')
    if counts['figure'] and figures_highres < counts['figure']:
        print('INFO: Some figures remain preview-only or missing; use --browser-figures')
        print('      (or --browser-assets to recover both tables and figures).')

if __name__=='__main__':main()
