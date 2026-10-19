import importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def module(relative):
    p=ROOT/relative
    name=relative.replace('/','_').replace('.','_')
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

class Dummy:
    def __init__(self):self.calls=[]
    def available_tools(self):return [{'name':'grep_page'},{'name':'read_context'}]
    def remaining_calls(self):return 8
    def call(self, tool, args):
        self.calls.append((tool,args))
        if tool=='grep_page':return {'accepted':True,'data':{'hits':[{'path':'authorized_issue.txt','offset':340,'text':'Revenue by Source'}], 'complete':True,'next_cursor':None}}
        if tool=='read_context':return {'accepted':True,'data':{'start_offset':100,'text':'Table: Revenue by Source\n[in millions of dollars]\n1940 | 7.0'}}
        raise ValueError('unsupported')

class Check(unittest.TestCase):
    def test_bundle(self):
        script=module('tools/inspect_bundle.py')
        script.main()
    def test_selection_bounded(self):
        s=module('integration/selection.py')
        for phase in ('locate','extract','compute','finalize'):
            names=s.select_names('What is the standard deviation of the series from 1951 to 1960?',phase)
            self.assertLessEqual(len(names),3)
            self.assertEqual(len(names),len(set(names)))
        self.assertEqual(s.category('What were the accepted Treasury tenders?'),'OA-Treasury')
        self.assertEqual(s.category('Calculate an ARIMA forecast'),'OA-Advanced')
        self.assertEqual(s.callable_programs('Time series from 1951 to 1960','compute')[0],'period_ledger.py')
    def test_scoped_actual_windows(self):
        m=module('programs/scoped_search.py');ctx=Dummy()
        r=m.run(ctx,{'query':'Revenue by Source','max_windows':1})
        self.assertEqual(r['status'],'ok');self.assertEqual(r['outputs']['windows'][0]['path'],'authorized_issue.txt')
        self.assertFalse(r['outputs']['table_header_bound'])
        self.assertIsNone(r['outputs']['target_periods_complete'])
        self.assertEqual(ctx.calls[-1][1]['path'],'authorized_issue.txt')
    def test_period_missing_then_complete(self):
        m=module('programs/period_ledger.py')
        req=['FY2001','FY2002','FY2003']
        def obs(p,v):return {'period':p,'value':v,'unit':'millions','source_ref':{'result_id':'real_'+p,'path':['data','text']}}
        a=m.run(None,{'required_periods':req,'observations':[obs('FY2001','1'),obs('FY2003','3')]})
        self.assertFalse(a['outputs']['complete']);self.assertEqual(a['outputs']['missing_periods'],['FY2002'])
        b=m.run(None,{'required_periods':req,'observations':[obs('FY2001','1'),obs('FY2002','2'),obs('FY2003','3')]})
        self.assertTrue(b['outputs']['complete']);self.assertEqual([o['period'] for o in b['outputs']['aligned_observations']],req)
    def test_period_conflict(self):
        m=module('programs/period_ledger.py')
        x={'period':'FY2010','value':'3','unit':'millions','source_ref':{'result_id':'r','path':['data','text']}}
        y={**x,'value':'4','source_ref':{'result_id':'r2','path':['data','text']}}
        a=m.run(None,{'required_periods':['FY2010'],'observations':[x,y]})
        self.assertFalse(a['outputs']['complete']);self.assertEqual(a['outputs']['conflicting_periods'],['FY2010'])
    def test_series_named_ratio(self):
        m=module('programs/series_metrics.py')
        a=m.run(None,{'operation':'ratio','numerator':200,'denominator':100,'output_percent':True,'unit':'percent','decimals':2})
        self.assertEqual(a['status'],'ok');self.assertEqual(a['outputs']['answer'],'200.00')
    def test_treasury_quote(self):
        m=module('programs/treasury_transform.py')
        a=m.run(None,{'operation':'price_32nds','quote':'99.27','quote_basis':'thirty_seconds','decimals':5})
        self.assertEqual(a['outputs']['answer'],'99.84375')
        self.assertEqual(m.run(None,{'operation':'price_32nds','quote':'99.27','quote_basis':'decimal'})['status'],'needs_input')
    def test_box_cox(self):
        m=module('programs/treasury_transform.py')
        a=m.run(None,{'operation':'box_cox','value':10,'lambda':1,'decimals':2})
        self.assertEqual(a['outputs']['answer'],'9.00')
    def test_units(self):
        m=module('programs/treasury_transform.py')
        a=m.run(None,{'operation':'unit_scale','value':'12.5','source_unit':'millions_of_dollars','target_unit':'dollars','decimals':0})
        self.assertEqual(a['outputs']['answer'],'12500000')
    def test_no_gold_id_table(self):
        bank=json.loads((ROOT/'bank/bank.json').read_text());txt=json.dumps(bank,ensure_ascii=False)
        import re
        self.assertIsNone(re.search(r'officeqa:UID\d+',txt))
        self.assertEqual(bank['status'],'candidate_not_measured')

if __name__=='__main__':unittest.main()
