/* Nonmodal, server-owned FIFO. Polling never changes the chosen job/page. */
'use strict';
(() => {
  const byId=id=>document.getElementById(id);
  let selected=null, jobs=[], signature='', previewURL='', submitting=0, requestSequence=0, appliedSequence=0;
  const url=(job,name)=>`/api/conversion-jobs/${encodeURIComponent(job.id)}/files/${name.split('/').map(encodeURIComponent).join('/')}`;
  function preview(){
    const job=jobs.find(j=>j.id===selected), pages=job?.pages||[], picker=byId('queue-page');
    const chosen=picker.value;
    if(picker.dataset.job!==selected || picker.options.length!==pages.length){
      picker.replaceChildren(...pages.map(p=>new Option(`第 ${p.number} 页 · 原文件第 ${p.physical_page} 页${p.layout_needs_review?' · 布局待复核':''}`,String(p.number))));
      if(picker.dataset.job===selected && pages.some(p=>String(p.number)===chosen))picker.value=chosen;
      picker.dataset.job=selected||'';
    }
    byId('queue-preview').hidden=!pages.length;
    const page=pages.find(p=>String(p.number)===picker.value);
    const next=page?url(job,page.preview):'';
    if(next!==previewURL){
      previewURL=next;
      if(next)byId('queue-frame').src=next;else byId('queue-frame').removeAttribute('src');
    }
  }
  function render(){
    const active=jobs.filter(j=>['queued','running'].includes(j.status)).length;
    byId('queue-count').textContent=active?`（${active} 份进行中 / 等待）`:'';
    const visible=jobs.filter(j=>['queued','running'].includes(j.status)||j.id===selected||jobs.indexOf(j)>=jobs.length-12);
    const next=JSON.stringify(visible);
    if(next!==signature){
      signature=next;
      const nodes=visible.map(job=>{
        const row=document.createElement('div');row.className='queue-item';
        const button=document.createElement('button');button.type='button';
        button.className='button secondary small';button.textContent=job.names.join('、');
        button.setAttribute('aria-pressed',String(job.id===selected));
        button.onclick=()=>{selected=job.id;signature='';render();};
        const status=document.createElement('span');
        const labels={queued:`排队第 ${job.queue_position} 份`,running:'转换中',completed:job.report?.unresolvedLayouts?.length?'已完成（部分页面布局待复核）':'已完成',failed:'失败',interrupted:'已中断','waiting-for-layout-resolution':'布局待解决（已保存）'};
        status.textContent=`${labels[job.status]} · 已完成 ${job.pages.length}${job.pages[0]?.total?'/'+job.pages[0].total:''} 页`;
        if(job.status==='running'&&job.progress)status.textContent+=` · ${job.progress.stage} ${job.progress.percent}%`;
        row.append(button,status);
        if(job.error){const error=document.createElement('p');error.textContent=job.error;row.append(error);}
        if(job.status==='completed'){
          const link=document.createElement('a');link.textContent='下载完整结果';link.href=url(job,job.result);link.download='';row.append(link);
        }
        return row;
      });
      byId('queue-items').replaceChildren(...nodes);
    }
    preview();
  }
  async function poll(){
    const sequence=++requestSequence;
    try{
      const response=await fetch('/api/conversion-jobs',{cache:'no-store'});
      if(!response.ok)throw new Error('无法读取转换队列');
      const result=await response.json();
      if(sequence<appliedSequence)return;
      appliedSequence=sequence;jobs=result;
      if(!selected)selected=(jobs.find(j=>j.status==='running')||jobs.at(-1))?.id||null;
      byId('queue-message').textContent=submitting?'正在上传新任务，可继续操作…':'后台按提交顺序处理；点击任务查看已完成页面。刷新网页不会中断转换。';
      render();
    }catch(error){byId('queue-message').textContent=`${error.message}，稍后自动重试；后台任务不受影响。`;}
  }
  window.conversionQueue={async submit(data){
    submitting++;
    try{
      const response=await fetch('/api/conversion-jobs',{method:'POST',body:data});
      if(response.status===404)throw new Error('此端口仍是旧服务，请重启本地服务后重试');
      if(!response.ok){const error=await response.json();throw new Error(error.detail||'加入队列失败');}
      const job=await response.json();selected=job.id;
      byId('conversion-queue').querySelector('details').open=true;
    }finally{submitting--;await poll();}
  }};
  byId('queue-page').onchange=preview;
  async function tick(){await poll();setTimeout(tick,1500);}
  tick();
})();
