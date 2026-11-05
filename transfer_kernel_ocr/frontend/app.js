/* Three public conversions; reconstruction runs in the server-owned queue. */
'use strict';
const $=selector=>document.querySelector(selector);
const MODES={
  html:{accept:'.pdf',action:'转换并下载 HTML ZIP',description:'读取可编辑 PDF 的原生文字、表格和图片，生成可编辑 HTML。'},
  md:{accept:'.pdf',action:'转换并下载 Markdown ZIP',description:'读取可编辑 PDF 的原生内容，生成 Markdown 与图片资源。'},
  pdf:{accept:'.pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp',action:'加入 PDF 转换队列',description:'本地识别图片或扫描 PDF，按源版面重建可编辑文字；可查看已完成页面。'}
};
const state={mode:'html',files:[],busy:false,resultURL:null};
const operationProgress=new OperationProgress();
let toastTimer;
function toast(message,error=false){const node=$('#toast');node.textContent=message;node.classList.toggle('error',error);node.classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.add('hidden'),6000);}
function accepted(file){return (state.mode==='pdf'?/\.(pdf|png|jpe?g|tiff?|webp|bmp)$/i:/\.pdf$/i).test(file.name);}
function pages(){
  const value=$('#page-range').value.trim().replace(/，/g,',').replace(/\s*([,-])\s*/g,'$1');
  if(!value)return '';
  if(!/^[1-9]\d*(?:-[1-9]\d*)?(?:,[1-9]\d*(?:-[1-9]\d*)?)*$/.test(value))throw new Error('页码请填写 1-3,5,8-10 这样的格式');
  for(const part of value.split(',')){const [start,end=start]=part.split('-').map(Number);if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||end<start)throw new Error('页码范围无效');}
  return value;
}
function validation(){if(state.files.length>30)return '每次最多30个文件';if(state.files.some(file=>!accepted(file)))return '请移除不支持当前转换方式的文件';try{pages();}catch(error){return error.message;}return '';}
function render(){
  const list=$('#pending-files');list.replaceChildren();
  state.files.forEach((file,index)=>{const row=document.createElement('div');row.className='pending-file';const name=document.createElement('span');name.textContent=file.name;const size=document.createElement('small');size.textContent=(file.size/1024/1024).toFixed(2)+' MB';const remove=document.createElement('button');remove.type='button';remove.textContent='移除';remove.disabled=state.busy;remove.setAttribute('aria-label','移除 '+file.name);remove.onclick=()=>{state.files.splice(index,1);render();};row.append(name,size,remove);list.append(row);});
  const error=validation();$('#import-error').textContent=error;$('#import-error').classList.toggle('hidden',!error);
  $('#parse-files').disabled=state.busy||!state.files.length||!!error;$('#parse-files').textContent=MODES[state.mode].action;
  $('#file-input').accept=MODES[state.mode].accept;$('#file-formats').textContent=state.mode==='pdf'?'PDF · PNG · JPG · TIFF · WebP · BMP · 每次最多30个文件':'PDF · 每次最多30个文件';$('#mode-description').textContent=MODES[state.mode].description;
}
function addFiles(files){if(state.busy)return;const incoming=[...files],valid=incoming.filter(accepted);state.files.push(...valid);if(valid.length!==incoming.length)toast('已跳过格式不支持的文件',true);render();}
function setBusy(value){state.busy=value;$('#busy-overlay').classList.toggle('hidden',!value);$('.app').inert=value;if(value)operationProgress.start();else operationProgress.stop();render();}
function filename(response,fallback){const header=response.headers.get('content-disposition')||'';const encoded=/filename\*\s*=\s*UTF-8'[^']*'([^;]+)/i.exec(header);let name;if(encoded){try{name=decodeURIComponent(encoded[1].trim());}catch{}}if(!name){const match=/filename\s*=\s*(?:"([^"]*)"|([^;]+))/i.exec(header);name=match&&(match[1]||match[2].trim());}return (name||fallback).replace(/[\\/:*?"<>|\u0000-\u001f]/g,'_');}
function download(url,name){const link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();}
async function submit(){
  if(state.busy||!state.files.length||validation())return;
  const files=[...state.files],mode=state.mode,data=new FormData();files.forEach(file=>data.append('files',file));data.append('pages',pages());
  if(mode==='pdf'){
    data.append('format','pdf');state.files=[];render();
    try{await window.conversionQueue.submit(data);toast('已加入队列，可继续提交文件或查看页面预览');}
    catch(error){state.files=[...files,...state.files];render();toast(error.message,true);}return;
  }
  data.append('format',mode);setBusy(true);
  try{
    const response=await operationProgress.request('/api/convert/reading',{method:'POST',body:data});
    if(!response.ok){const error=await response.json();throw new Error(error.detail||'转换失败');}
    const blob=await operationProgress.download(response);
    if(state.resultURL)URL.revokeObjectURL(state.resultURL);
    state.resultURL=URL.createObjectURL(blob);const name=filename(response,`PDF-${mode}.zip`);
    $('#reading-download').href=state.resultURL;$('#reading-download').download=name;$('#result-description').textContent='已生成文档、图片资源与源 PDF 的 ZIP 包。';$('#reading-result').classList.remove('hidden');
    download(state.resultURL,name);state.files=[];
  }catch(error){toast(error.message,true);}finally{setBusy(false);}
}
document.querySelectorAll('input[name="mode"]').forEach(input=>input.onchange=()=>{state.mode=input.value;render();});
$('#choose-files').onclick=event=>{event.stopPropagation();$('#file-input').click();};
$('#file-input').onchange=event=>{addFiles(event.target.files);event.target.value='';};
$('#dropzone').onclick=()=>{if(!state.busy)$('#file-input').click();};
$('#dropzone').onkeydown=event=>{if(event.target===event.currentTarget&&['Enter',' '].includes(event.key)){event.preventDefault();$('#file-input').click();}};
for(const event of ['dragover','dragenter'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.add('dragover');});
for(const event of ['dragleave','drop'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.remove('dragover');});
$('#dropzone').addEventListener('drop',event=>addFiles(event.dataTransfer.files));
$('#parse-files').onclick=submit;$('#page-range').oninput=render;
window.addEventListener('beforeunload',()=>{if(state.resultURL)URL.revokeObjectURL(state.resultURL);});
render();
fetch('/api/health').then(response=>response.json()).then(data=>{$('#ocr-status').textContent=data.ocr.available?'本地 OCR 引擎已就绪':'原生 PDF 可转换 · 扫描转换请先完成模型安装';}).catch(()=>{$('#ocr-status').textContent='本地服务暂不可用，请检查终端';});
