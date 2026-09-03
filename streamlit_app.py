import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, scipy.sparse as sp
import streamlit as st

DEFAULT_ROOT="/content/drive/MyDrive/ecommerce_recommender"
st.set_page_config(page_title="Discover • Recommender",page_icon="🛍️",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px}.hero{padding:2rem;border-radius:20px;background:linear-gradient(120deg,#312e81,#0f766e);color:white;margin-bottom:1.3rem}.hero h1{margin:0}.hero p{opacity:.88;margin:.5rem 0 0}.card{background:#161b2c;border:1px solid #30384f;padding:1rem;border-radius:14px}.model-note{border-left:4px solid #2dd4bf;background:#102a32;padding:.8rem 1rem;border-radius:8px;margin:.75rem 0}</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading offline recommendation artifacts…")
def load(root_text):
 root=Path(root_text);m=json.loads((root/"artifacts/inference_manifest.json").read_text());maps=joblib.load(root/m["mappings"])
 c=pd.read_parquet(root/m["catalog"]);c.product_id=c.product_id.astype(str)
 return {"c":c,"p":pd.read_parquet(root/m["popularity"]),"item":joblib.load(root/m["item_cf"]),"als":joblib.load(root/m["als"]),"ui":sp.load_npz(root/m["ui"]),"u":pd.Index(maps["users"],dtype="string"),"i":pd.Index(maps["items"],dtype="string"),"v":joblib.load(root/m["tfidf"]),"nn":joblib.load(root/m["content_nn"]),"ids":list(map(str,joblib.load(root/m["content_ids"]))),"cfg":json.loads((root/m["hybrid_config"]).read_text())}
def enrich(a,d):
 return d.merge(a["c"][[x for x in ["product_id","title","category","average_rating"] if x in a["c"]]],on="product_id",how="left")
def cat_filter(d,cat):
 return d if cat=="All categories" else d[d.category.astype(str).str.casefold()==cat.casefold()]
def popular(a,cat,k):
 return cat_filter(a["p"],cat).head(k).copy()
def also_interacted(a,pid,cat,k):
 q=np.flatnonzero(a["i"]==str(pid))
 if not len(q):return pd.DataFrame()
 z,s=a["item"].similar_items(int(q[0]),N=min(k*30,len(a["i"])))
 d=enrich(a,pd.DataFrame({"product_id":a["i"].take(z).astype(str),"score":s}));return cat_filter(d[d.product_id!=str(pid)],cat).head(k)
def content(a,pid,cat,k):
 if str(pid) not in a["ids"]:return pd.DataFrame()
 source=a["c"][a["c"].product_id==str(pid)]
 if source.empty:return pd.DataFrame()
 dis,pos=a["nn"].kneighbors(a["v"].transform([str(source.iloc[0].get("product_text",""))]),n_neighbors=min(k*30,len(a["ids"])))
 d=enrich(a,pd.DataFrame({"product_id":np.asarray(a["ids"])[pos[0]],"score":1-dis[0]}));return cat_filter(d[d.product_id!=str(pid)],cat).head(k)
def personal(a,uid,k):
 q=np.flatnonzero(a["u"]==str(uid))
 if not len(q):return pd.DataFrame()
 r=int(q[0]);z,s=a["als"].recommend(r,a["ui"][r],N=k,filter_already_liked_items=True)
 return enrich(a,pd.DataFrame({"product_id":a["i"].take(z).astype(str),"score":s}))
def norm(d):
 d=d[["product_id","score"]].copy();x=d.score.astype(float);d.score=(x-x.min())/(x.max()-x.min()+1e-12);return d
def hybrid(a,uid,pid,cat,k):
 parts=[];cfg=a["cfg"];n=cfg.get("candidate_k",100)
 if uid:
  d=personal(a,uid,n)
  if not d.empty:parts.append(norm(d).assign(score=lambda x:x.score*cfg["als"]))
 if pid:
  d=content(a,pid,cat,n)
  if not d.empty:parts.append(norm(d).assign(score=lambda x:x.score*cfg["similar_or_content"]))
 d=popular(a,cat,n).rename(columns={"popularity_score":"score"})
 if not d.empty:parts.append(norm(d).assign(score=lambda x:x.score*cfg["popularity"]))
 if not parts:return pd.DataFrame()
 return enrich(a,pd.concat(parts).groupby("product_id",as_index=False).score.sum().nlargest(k,"score"))

st.markdown("""<div class="hero"><h1>🛍️ Discover products you’ll love</h1><p>Explore popularity, product similarity, customer behaviour, and personalised suggestions.</p></div>""",unsafe_allow_html=True)
with st.sidebar:
 st.header("Recommendation controls")
 root=st.text_input("Artifact directory",DEFAULT_ROOT)
try:a=load(root)
except Exception as e:st.error(f"Could not load artifacts: {e}");st.stop()
methods={"Popular right now":"Popularity baseline","Similar products":"Semantic TF-IDF similarity","Customers also interacted with":"Behavioural item-CF association","For you":"Personalised ALS","Balanced discovery":"Hybrid ranking"}
choice=st.sidebar.radio("How would you like to explore?",list(methods),help="Each method uses a different recommendation signal.")
st.sidebar.caption(methods[choice])
k=st.sidebar.slider("How many results?",5,30,10)
categories=["All categories"]+sorted(a["c"].category.dropna().astype(str).unique())
cat=st.sidebar.selectbox("Category",categories,help="Restricts the source product and results where applicable.")
catalog=cat_filter(a["c"],cat).copy();catalog["label"]=catalog.title.fillna("Untitled")+" — "+catalog.product_id
pid=None
if choice in ["Similar products","Customers also interacted with","Balanced discovery"]:
 label=st.selectbox("Choose a source product",catalog.label);pid=str(catalog.loc[catalog.label==label,"product_id"].iloc[0])
uid=st.text_input("User ID",placeholder="Required only for For you; optional for Balanced discovery") if choice in ["For you","Balanced discovery"] else None
notes={"Popular right now":"Popular products are ranked by interaction volume.","Similar products":"Uses product text (title, features, description, and category). Results are semantically similar to the selected product.","Customers also interacted with":"Behavioural association only: these are products reviewed by overlapping users. They are not necessarily substitutes or semantically alike.","For you":"Uses sparse ALS collaborative filtering and excludes items the user has already interacted with.","Balanced discovery":"Combines personal, semantic-content, and popularity signals after score normalization."}
st.markdown(f'<div class="model-note"><b>How this works</b><br>{notes[choice]}</div>',unsafe_allow_html=True)
if st.button("Find recommendations",type="primary",use_container_width=True):
 if choice=="Popular right now":out=popular(a,cat,k)
 elif choice=="Similar products":out=content(a,pid,cat,k)
 elif choice=="Customers also interacted with":out=also_interacted(a,pid,cat,k)
 elif choice=="For you":out=personal(a,uid,k)
 else:out=hybrid(a,uid,pid,cat,k)
 if out.empty:st.warning("No recommendations matched. Try All categories, another source product, or Popular right now.")
 else:
  st.subheader(f"Top {len(out)} recommendations")
  cols=["title","category","average_rating","score","product_id"];st.dataframe(out[[x for x in cols if x in out]],hide_index=True,use_container_width=True,column_config={"score":st.column_config.NumberColumn("Recommendation score",format="%.3f"),"average_rating":st.column_config.NumberColumn("Rating",format="%.1f ⭐")})
