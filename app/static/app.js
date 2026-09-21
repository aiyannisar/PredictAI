const forms={
house:[
["area_sqft","Area (sq ft)","number",450,4200,1500],
["bedrooms","Bedrooms","number",1,6,3],
["bathrooms","Bathrooms","number",1,6,2],
["age_years","Property age (years)","number",0,30,5],
["distance_km","Distance to center (km)","number",0.5,25,7],
["location","Location tier","select",["Prime","Urban","Suburban","Outer"],"Urban"]
],
loan:[
["age","Age","number",20,70,32],["income","Annual income (₹)","number",18000,240000,85000],
["loan_amount","Loan amount (₹)","number",10000,1500000,400000],
["credit_score","Credit score","number",420,850,690],
["debt_to_income","Debt-to-income","number",0.05,.75,.28],
["employment_years","Employment (years)","number",0,25,5],
["previous_defaults","Previous defaults","number",0,3,0]
],
churn:[
["tenure_months","Tenure (months)","number",1,72,18],
["monthly_charges","Monthly charges (₹)","number",20,180,79],
["support_tickets","Support tickets","number",0,10,3],
["contract","Contract","select",["Month-to-month","One year","Two year"],"Month-to-month"],
["internet_service","Internet service","select",["Fiber","DSL","None"],"Fiber"],
["payment_method","Payment method","select",["Electronic","Card","Bank transfer"],"Electronic"]
],
maintenance:[
["temperature","Temperature (°C)","number",30,110,72],
["vibration","Vibration","number",0.05,1.5,.55],
["pressure","Pressure","number",10,60,34],
["rpm","RPM","number",800,3000,1850],
["operating_hours","Operating hours","number",50,12000,4200],
["load_factor","Load factor","number",.1,1,.72]
]};

let currentModule=null,currentPayload=null,currentPrediction=null;
const $=s=>document.querySelector(s);

function renderForm(key){
 currentModule=key;
 const f=$("#predictionForm"); f.innerHTML="";
 const labels={house:"Property Price",loan:"Loan Risk",churn:"Customer Churn",maintenance:"Machine Failure"};
 $("#modelName").textContent=labels[key];
 $("#workspaceSub").textContent="Tune the variables, then run local inference";
 forms[key].forEach(x=>{
   const [name,label,type,a,b,val]=x;
   const d=document.createElement("div"); d.className="field";
   const lab=document.createElement("label"); lab.textContent=label; d.appendChild(lab);
   if(type==="select"){
     const s=document.createElement("select");s.name=name;
     a.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;if(v===val)o.selected=true;s.appendChild(o)});
     d.appendChild(s);
   }else{
     const i=document.createElement("input");i.name=name;i.type=type;i.min=a;i.max=b;i.step=String(val).includes(".")?".01":"1";i.value=val;d.appendChild(i);
   }
   f.appendChild(d);
 });
 $("#workspace").classList.remove("hidden");
 $("#workspace").scrollIntoView({behavior:"smooth",block:"start"});
}

document.querySelectorAll(".module").forEach(b=>b.addEventListener("click",()=>renderForm(b.dataset.module)));

function readPayload(){
 const out={};new FormData($("#predictionForm")).forEach((v,k)=>{
   const el=document.querySelector(`[name="${k}"]`);
   out[k]=el.tagName==="SELECT"?v:Number(v);
 });return out;
}

$("#predictBtn").addEventListener("click",async()=>{
 const btn=$("#predictBtn");btn.disabled=true;btn.innerHTML="RUNNING MODEL <span>◌</span>";
 await new Promise(r=>setTimeout(r,650));
 currentPayload=readPayload();
 try{
  const r=await fetch(`/api/predict/${currentModule}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(currentPayload)});
  const data=await r.json(); if(!data.ok)throw Error(data.error);
  currentPrediction=data.result; showResult(data.result);
 }catch(e){alert(e.message)}
 btn.disabled=false;btn.innerHTML="RUN PREDICTION <span>→</span>";
});

function showResult(r){
 $("#result").classList.remove("hidden");$("#result").scrollIntoView({behavior:"smooth"});
 $("#resultLabel").textContent=r.label;$("#resultValue").textContent=r.headline;
 const pill=$("#riskPill");
 if(r.risk===null){pill.textContent="REGRESSION ESTIMATE";pill.style.color="var(--accent)"}
 else {const p=r.risk; pill.textContent=p>=.7?"HIGH RISK":p>=.4?"MEDIUM RISK":"LOW RISK";pill.style.color=p>=.7?"var(--danger)":p>=.4?"#ffd37a":"var(--accent)"}
 $("#drivers").innerHTML="<div class='eyebrow'>TOP MODEL DRIVERS</div>"+r.drivers.map(d=>`<div class="driver"><span>${d[0]}</span><div class="bar"><i style="width:${d[1]}%"></i></div><b>${d[1]}%</b></div>`).join("");
 $("#aiText").textContent="Gemini can interpret this prediction and explain the important drivers.";
}

$("#aiBtn").addEventListener("click",async()=>{
 const b=$("#aiBtn"),t=$("#aiText");b.disabled=true;b.textContent="✦ ANALYZING...";
 t.textContent="Gemini is reading the prediction context…";
 try{
  const r=await fetch("/api/ai",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({module:currentModule,payload:currentPayload,prediction:currentPrediction})});
  const d=await r.json();t.textContent=d.text||"No response.";
 }catch(e){t.textContent="Could not reach the AI layer. Check your local server and Gemini configuration."}
 b.disabled=false;b.textContent="✦ ASK GEMINI";
});

setTimeout(()=>{$("#bootStatus").textContent="LOADING LOCAL MODELS...";},500);
setTimeout(()=>{const b=$("#boot");b.style.opacity="0";setTimeout(()=>b.remove(),650)},1700);
