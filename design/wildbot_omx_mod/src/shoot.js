// usage: node shoot.js out.png "query"
const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch({args:['--use-gl=swiftshader','--enable-webgl','--ignore-gpu-blocklist','--allow-file-access-from-files']});
const p=await b.newPage({viewport:{width:1400,height:1000}});
p.on('console',m=>{if(m.type()==='error')console.log('ERR',m.text())});
const jobs=JSON.parse(process.argv[2]);
const http=require('http'),fs=require('fs'),path=require('path');
const srv=http.createServer((q,res)=>{const f=path.join('/home/claude/mod',decodeURIComponent(q.url.split('?')[0]));fs.readFile(f,(e,d)=>{if(e){res.writeHead(404);res.end();return;}res.writeHead(200,{'Content-Type':f.endsWith('.js')?'text/javascript':f.endsWith('.html')?'text/html':'application/octet-stream'});res.end(d);});}).listen(8765);
for(const [out,qs] of jobs){await p.goto('http://localhost:8765/src/render.html?'+qs);await p.waitForFunction(()=>document.title==='done',null,{timeout:120000});
const c=await p.$('canvas');await c.screenshot({path:out});console.log('wrote',out);}
await b.close();srv.close();})();
