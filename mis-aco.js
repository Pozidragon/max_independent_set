let V = [1,2,3,4];
let E = [[1,2],[2,3],[3,4]];


const adj = new Map(); V.forEach(v=>adj.set(v, new Set()));
E.forEach(([u,v])=>{ adj.get(u).add(v); adj.get(v).add(u); });
const deg = new Map(V.map(v=>[v, adj.get(v).size]));
const eta = new Map(V.map(v=>[v, 1.0/(deg.get(v)+1)]));


const nAnts=20, nIters=150, alpha=1.0, beta=2.0, rho=0.1, Q=1.0, tauMax=1.0, tauMin=0.01;
const tau = new Map(V.map(v=>[v, tauMax]));


function feasibleVertices(chosen, available){
    const forb = new Set(chosen);
    for(const u of chosen) for(const w of adj.get(u)) forb.add(w);
    return available.filter(v=>!forb.has(v));
}


function sampleProportional(items, weights){
    const total = weights.reduce((a,b)=>a+b,0);
    if(total===0) return items[Math.floor(Math.random()*items.length)];
    let r=Math.random()*total, acc=0;
    for(let i=0;i<items.length;i++){ acc+=weights[i]; if(acc>=r) return items[i]; }
    return items[items.length-1];
}


function constructSolution(){
    const chosen = new Set();
    let available = [...V];
    while(true){
        const feas = feasibleVertices(chosen, available);
        if(feas.length===0) break;
        const weights = feas.map(v=> Math.pow(tau.get(v), alpha) * Math.pow(eta.get(v), beta));
        const v = sampleProportional(feas, weights);
        chosen.add(v);
        const banned = new Set([v, ...adj.get(v)]);
        available = available.filter(x=>!banned.has(x));
    }
    return chosen;
}


function updatePheromones(bestGlobal){
    for(const v of V){ tau.set(v, Math.max(tauMin, tau.get(v)*(1-rho))); }
    const delta = Q/Math.max(1, bestGlobal.size);
    for(const v of bestGlobal){ tau.set(v, Math.min(tauMax, tau.get(v)+delta)); }
}


let bestSet=new Set(), bestVal=0;
for(let it=1; it<=nIters; it++){
    let iterBest=new Set(), iterBestVal=-1;
    for(let k=0;k<nAnts;k++){
        const S = constructSolution();
        if(S.size>iterBestVal){ iterBestVal=S.size; iterBest=S; }
    }
    if(iterBestVal>bestVal){ bestVal=iterBestVal; bestSet=new Set(iterBest); }
    updatePheromones(bestSet);
    if(it===1 || it % Math.max(1, Math.floor(nIters/10))===0){
        console.log(`[iter ${it}] best_so_far=${bestVal}, best_set=[${[...bestSet].sort().join(', ')}]`);
    }
}


console.log('\nГлобально найкраще:');
console.log('S* =', [...bestSet].sort(), '|S*| =', bestVal);
