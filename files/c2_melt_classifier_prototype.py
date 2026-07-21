import json,sys,os,re,collections
import pandas as pd, numpy as np
props={r["candidate_id"]:r for r in (json.loads(l) for l in open(sys.argv[1]))}
reviews=[json.loads(l) for l in open(sys.argv[2])]

AXIS=("time","day","hour","min","sec","conc","dose","distance","wavelength","temp",
      "week","month","age","cycle","freq","voltage","current","position","depth","ph ",
      "passage","generation","branchpoint","length","ratio","mass","weight","diameter","angle")
INDEX=("replicate","sample","animal","id","index","no.","subject","cell","clone","mouse","rep")

def colkind(name,ser):
    n=str(name).lower().strip()
    if any(k in n for k in INDEX) and ser.dropna().map(lambda v:isinstance(v,(int,np.integer)) or (isinstance(v,float) and float(v).is_integer())).all():
        return "index"
    if ser.dtype==object or str(ser.dtype).startswith("str"): return "label"
    return "measure"

def axislike(name): 
    n=str(name).lower()
    return any(k in n for k in AXIS)

def classify_independent(df):
    cols=list(df.columns)
    kinds={c:colkind(c,df[c]) for c in cols}
    labels=[c for c in cols if kinds[c]=="label"]
    measures=[c for c in cols if kinds[c]=="measure"]
    idxs=[c for c in cols if kinds[c]=="index"]
    # axis-like measure column (controlled independent variable)
    axis_measures=[c for c in measures if axislike(c)]
    # rule order
    if labels:  # a categorical grouping column present -> bar (grouped comparison)
        return "bar","x=%s(categorical)"%labels[0]
    # no label col. All numeric.
    nonaxis_measures=[c for c in measures if not axislike(c)]
    if len(measures)>=2 and len(axis_measures)==0 and not idxs:
        # wide grouped: headers are group names (genotypes) OR two measurements correlated
        # heuristic: >=3 numeric cols with non-axis headers -> melt to categorical scatter
        if len(measures)>=3: return "scatter","melt-wide-%dgroups"%len(measures)
        # exactly 2 numeric measurement cols, neither axis-like -> correlation scatter
        return "scatter","two-measure-correlation"
    if axis_measures and len(measures)>=2:
        # real ordered independent var + response -> line, unless x has heavy duplicates (scatter)
        xc=axis_measures[0]; xv=df[xc].dropna()
        if xv.duplicated().mean()>0.3: return "scatter","axis-x-with-duplicates"
        return "line","axis-x=%s"%xc
    if len(measures)==1 and idxs:
        return "bar","index-x-single-measure"
    return "unknown","cols=%s"%[kinds[c] for c in cols]

def consensus(r):
    fams=[str((m.get("output") or {}).get("chart_family")).lower() for m in r.get("model_reviews",[]) if isinstance(m.get("output"),dict) and (m.get("output") or {}).get("chart_family")]
    return fams[0] if len(fams)==2 and fams[0]==fams[1] else None

rej=[r for r in reviews if r["proposal_type"]=="single_panel" and r["status"]=="rejected"]
conf=collections.Counter(); match=0; total=0; readerr=0; details=[]
for r in rej:
    c=consensus(r)
    if c is None: continue
    p=props[r["candidate_id"]]; st=p["source_table"]; path=st["path"]
    if not os.path.exists(path): readerr+=1; continue
    try:
        df=pd.read_excel(path,sheet_name=st.get("sheet_name"),header=0)
    except Exception: readerr+=1; continue
    pred,why=classify_independent(df)
    total+=1; conf[(c,pred)]+=1
    if pred==c: match+=1
    details.append((r["candidate_id"][:45],c,pred,why))
print("=== INDEPENDENT MELT CLASSIFIER vs JUDGE CONSENSUS (agree-rejects) ===")
print("total evaluated=%d  readerr=%d"%(total,readerr))
print("MATCH (independent==consensus) = %d/%d = %.1f%%"%(match,total,100*match/total if total else 0))
print("\nConfusion (consensus -> predicted): count")
for (c,pred),v in sorted(conf.items(),key=lambda x:-x[1]): print("  %3d  consensus=%-8s independent=%-8s"%(v,c,pred))
print("\nSample 12:")
for d in details[:12]: print("  %-46s cons=%-8s pred=%-8s %s"%d)

# --- refinement: binding-class equivalence (grouped={bar,scatter} vs line) ---
def bindclass(f): return "grouped" if f in ("bar","scatter") else f
bind_match=0; bind_total=0; unknown=0
grp_conf=collections.Counter()
for r in rej:
    c=consensus(r)
    if c is None: continue
    p=props[r["candidate_id"]]; st=p["source_table"]; path=st["path"]
    if not os.path.exists(path): continue
    try: df=pd.read_excel(path,sheet_name=st.get("sheet_name"),header=0)
    except Exception: continue
    pred,why=classify_independent(df)
    bind_total+=1
    if pred=="unknown": unknown+=1; continue
    grp_conf[(bindclass(c),bindclass(pred))]+=1
    if bindclass(pred)==bindclass(c): bind_match+=1
print("\n=== BINDING-CLASS (grouped={bar,scatter} vs line) equivalence ===")
print("binding-class match = %d/%d = %.1f%% (unknown=%d)"%(bind_match,bind_total,100*bind_match/bind_total,unknown))
for (c,pred),v in sorted(grp_conf.items(),key=lambda x:-x[1]): print("  %3d  cons=%-8s pred=%-8s"%(v,c,pred))
