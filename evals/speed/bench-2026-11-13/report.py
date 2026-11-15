import json,sys
src,want=sys.argv[1],sys.argv[2:]
R={}
for l in open(src):
    d=json.loads(l)
    m,e,w=d.get("model"),d.get("engine"),d.get("work")
    if want and not any(x in m for x in want): continue
    if d.get("error"): print("ERR",m,e,w,str(d["error"])[:140]); continue
    if w=="warmup": continue
    if w=="load": R.setdefault(m,{}).setdefault("load",{})[e]={"load":d["load_s"]}; continue
    rows=d["rows"]; r=rows[0]
    st=[s for x in rows for s in x["steps"]]
    rd=sum(s.get("read") or 0 for s in st); rs=sum(s.get("read_s") or 0 for s in st); g=sum(s.get("gen") or 0 for s in st); gs=sum(s.get("gen_s") or 0 for s in st)
    later=rows[1:] if w=="multiturn" else rows
    t=(r.get("text") or "").replace(",","").lower()
    right={"long":r.get("recall"),"large":r.get("recall"),"xlarge":r.get("recall"),"tools":int("26742.25" in t or "26742.2" in t),"skill":int("78" in t and bool(r.get("wrote"))),"longtools":int("inventory-sync" in t),"multiturn":round(sum(bool(x.get("right")) for x in rows)/len(rows),2)}.get(w)
    R.setdefault(m,{}).setdefault(w,{})[e]=dict(ttft=(lambda v: round(sum(v)/len(v),2) if v else None)([x["ttft"] for x in later if x.get("ttft") is not None]),total=round(sum(x.get("total") or 0 for x in rows),1),read=round(rd/rs) if rs else None,gen=round(g/gs,1) if gs else None,gb=round(d["peak_total"]/2**30,1),right=right,err="; ".join((x.get("error") or "")[:50] for x in rows if x.get("error")))
for m,W in R.items():
    print("==",m)
    for w,E in W.items():
        for e in ("vireo","llama","mlx"):
            if e in E: print(f"  {w:10} {e:6}",E[e])
