const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../frontend/progress.js'),'utf8');

function setup(fetch){
  const nodes=new Map(),timeouts=new Map();let sequence=0,now=0;
  const node=selector=>{if(!nodes.has(selector))nodes.set(selector,{textContent:'',value:null,removeAttribute(){this.value=null;}});return nodes.get(selector);};
  const context={document:{querySelector:node},fetch,Headers,Response,Blob,AbortController,DOMException,
    performance:{now:()=>now},setInterval:()=>++sequence,clearInterval(){},
    setTimeout(callback,delay){const id=++sequence;if(delay===180)queueMicrotask(callback);else timeouts.set(id,callback);return id;},
    clearTimeout(id){timeouts.delete(id);}};
  vm.runInNewContext(source+'\nglobalThis.Controller=OperationProgress;',context);
  const controller=new context.Controller();
  return {controller,node,timeouts,advance(value){now=value;}};
}

test('waiting time does not invent a percentage and stop resets the next task',()=>{
  const {controller,node,advance}=setup();controller.start();advance(90000);
  assert.equal(node('#busy-progress').value,null);
  assert.equal(node('#busy-percent').textContent,'准备中');
  const previous=controller.run;
  controller.show(previous,42,'匹配文字字体','第 2 / 8 页');
  controller.show(previous,10,'下个阶段');
  assert.equal(node('#busy-progress').value,42);
  controller.stop();assert.equal(previous.controller.signal.aborted,true);
  controller.start();controller.show(previous,100,'旧任务');
  assert.equal(node('#busy-progress').value,null);controller.stop();
});

test('tracked POST registers a progress token and preserves form data and headers',async()=>{
  const calls=[];
  const {controller}=setup(async(url,options)=>{calls.push({url,options});
    if(url==='/api/progress')return Response.json({id:'token'});
    if(url==='/api/progress/token')return Response.json({status:'completed',percent:100,stage:'已生成'});
    return new Response('result');});
  controller.start();const body={example:true};
  const response=await controller.request('/api/convert/rebuild',{method:'POST',body,headers:{'X-Example':'yes'}});
  assert.equal(await response.text(),'result');
  const main=calls.find(call=>call.url==='/api/convert/rebuild');
  assert.equal(main.options.body,body);
  assert.equal(main.options.headers.get('x-progress-id'),'token');
  assert.equal(main.options.headers.get('x-example'),'yes');controller.stop();
});

test('progress service failure leaves the actual conversion available',async()=>{
  const {controller,node}=setup(async url=>url==='/api/progress'?new Response('unavailable',{status:503}):new Response('converted'));
  controller.start();const response=await controller.request('/api/documents',{method:'POST'});
  assert.equal(await response.text(),'converted');
  assert.match(node('#busy-progress-detail').textContent,/任务继续处理/);controller.stop();
});

test('file progress reaches 100 only after all response bytes have arrived',async()=>{
  const {controller,node}=setup();controller.start();
  const response=new Response(new Uint8Array([1,2,3,4]),{headers:{'content-length':'4','content-type':'application/pdf'}});
  const blob=await controller.download(response);
  assert.deepEqual([...new Uint8Array(await blob.arrayBuffer())],[1,2,3,4]);
  assert.equal(blob.type,'application/pdf');
  assert.equal(node('#busy-progress').value,100);
  assert.equal(controller.run.receiving,true);controller.stop();
});

test('a late response from an older poll cannot overwrite the new progress',async()=>{
  let resolve;
  const {controller,node}=setup(()=>new Promise(done=>{resolve=done;}));
  controller.start();const previous=controller.run;previous.id='old';
  const pending=controller.poll(previous);
  controller.stop();controller.start();
  resolve(Response.json({status:'completed',percent:100,stage:'旧任务'}));await pending;
  assert.equal(node('#busy-progress').value,null);
  assert.equal(node('#busy-stage').textContent,'准备处理');controller.stop();
});

test('server completion stays below 100 until the result has been received',async()=>{
  const {controller,node}=setup(async()=>Response.json({status:'completed',percent:100,stage:'结果已生成'}));
  controller.start();controller.run.id='done';await controller.poll(controller.run);
  assert.equal(node('#busy-progress').value,95);controller.stop();
});

test('parsed JSON completion also reaches 100 and closes its progress polling',async()=>{
  const {controller,node}=setup();controller.start();
  await controller.complete('解析完成','文档数据已接收');
  assert.equal(node('#busy-progress').value,100);
  assert.equal(node('#busy-stage').textContent,'解析完成');
  assert.equal(controller.run.controller.signal.aborted,true);
  const previous=controller.run;controller.stop();controller.start();
  await controller.complete('旧结果','',previous);
  assert.equal(node('#busy-progress').value,null);controller.stop();
});

test('a late polling failure cannot replace completed result details',async()=>{
  let resolve;
  const {controller,node}=setup(()=>new Promise(done=>{resolve=done;}));
  controller.start();controller.run.id='current';
  const pending=controller.poll(controller.run);
  await controller.complete('解析完成','文档数据已接收');
  resolve(new Response('unavailable',{status:503}));await pending;
  assert.equal(node('#busy-progress-detail').textContent,'文档数据已接收');
  assert.equal(node('#busy-progress').value,100);controller.stop();
});
