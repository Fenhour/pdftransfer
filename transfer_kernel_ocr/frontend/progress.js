/* Actual backend work milestones and response bytes; never a simulated timer. */
'use strict';
class OperationProgress {
  constructor(){this.run=null;}
  start(){
    this.stop();
    const run={started:performance.now(),controller:new AbortController(),timer:null,pollTimer:null,percent:0};
    this.run=run;
    document.querySelector('#busy-progress').removeAttribute('value');
    document.querySelector('#busy-percent').textContent='准备中';
    document.querySelector('#busy-stage').textContent='准备处理';
    document.querySelector('#busy-progress-detail').textContent='正在准备请求，请保持页面打开。';
    const elapsed=()=>{const seconds=Math.floor((performance.now()-run.started)/1000);document.querySelector('#busy-elapsed').textContent=`已用时 ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;};
    elapsed();run.timer=setInterval(elapsed,1000);
  }
  stop(){const run=this.run;if(!run)return;this.run=null;clearInterval(run.timer);clearTimeout(run.pollTimer);run.controller.abort();}
  show(run,percent,stage,detail=''){
    if(this.run!==run)return;
    if(Number.isFinite(percent)){run.percent=Math.max(run.percent,Math.min(100,percent));document.querySelector('#busy-progress').value=run.percent;document.querySelector('#busy-percent').textContent=`${Math.floor(run.percent)}%`;}
    document.querySelector('#busy-stage').textContent=stage;
    document.querySelector('#busy-progress-detail').textContent=detail;
  }
  async poll(run){
    if(this.run!==run||run.receiving)return;
    let terminal=false;
    try{
      const response=await fetch(`/api/progress/${run.id}`,{cache:'no-store',signal:run.controller.signal});
      if(!response.ok)throw new Error('Progress unavailable');
      const data=await response.json();
      if(this.run!==run||run.receiving)return;
      const count=data.total?`（${data.current??0} / ${data.total}）`:'';
      this.show(run,Math.min(95,data.percent*.95),data.stage,`${data.message||''}${count}`);
      terminal=data.status==='completed'||data.status==='failed';
    }catch(error){if(this.run===run&&!run.receiving&&error.name!=='AbortError')document.querySelector('#busy-progress-detail').textContent='进度连接暂时中断，任务仍在处理，正在重连…';}
    if(this.run===run&&!run.receiving&&!terminal)run.pollTimer=setTimeout(()=>this.poll(run),500);
  }
  async request(url,options={}){
    const run=this.run;
    const tracked=run&&options.method==='POST'&&(/^\/api\/convert\//.test(url)||url==='/api/documents'||url==='/api/detection-preview'||url==='/api/demo'||/\/export\//.test(url));
    if(!tracked)return fetch(url,options);
    const headers=new Headers(options.headers);
    try{
      const created=await fetch('/api/progress',{method:'POST',signal:run.controller.signal});
      if(!created.ok)throw new Error('Progress unavailable');
      const data=await created.json();
      if(this.run!==run)throw new DOMException('Operation ended','AbortError');
      run.id=data.id;headers.set('X-Progress-ID',run.id);
      this.show(run,0,'发送请求','正在上传文件或提交当前文档。');
      void this.poll(run);
    }catch(error){if(error.name==='AbortError')throw error;this.show(run,null,'正在处理','实时进度暂不可用，任务继续处理。');}
    return fetch(url,{...options,headers});
  }
  async complete(stage='处理完成',detail='文档数据已接收。',run=this.run){
    if(!run||this.run!==run)return;
    run.receiving=true;clearTimeout(run.pollTimer);run.controller.abort();
    this.show(run,100,stage,detail);
    await new Promise(resolve=>setTimeout(resolve,180));
  }
  async download(response){
    const run=this.run,total=Number(response.headers.get('content-length'));
    if(!run||!response.body?.getReader)return response.blob();
    run.receiving=true;clearTimeout(run.pollTimer);run.controller.abort();
    const reader=response.body.getReader(),chunks=[];let received=0;
    this.show(run,95,'接收结果文件',total?`0 / ${(total/1024/1024).toFixed(1)} MB`:'正在接收文件…');
    while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.byteLength;this.show(run,total?95+Math.min(1,received/total)*5:null,'接收结果文件',`${(received/1024/1024).toFixed(1)}${total?' / '+(total/1024/1024).toFixed(1):''} MB`);}
    await this.complete('处理完成','文件已接收，正在发起下载。',run);
    return new Blob(chunks,{type:response.headers.get('content-type')||'application/octet-stream'});
  }
}
